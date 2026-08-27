import os
from dotenv import load_dotenv
load_dotenv()
import uuid

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from prompts import SPEZI_SYSTEM_PROMPT
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

try:
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
except ModuleNotFoundError:
    try:
        from langchain_postgres.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
    except ModuleNotFoundError:
        from langchain_postgres import HybridSearchConfig, reciprocal_rank_fusion

# Cloud Database Configuration (Neon)
# Grab the Neon connection string from environment variables
NEON_URI = os.getenv("NEON_DB_URI")
if not NEON_URI:
    raise ValueError("NEON_DB_URI environment variable is missing!")

# LangChain's PGEngine requires the sqlalchemy prefix
DB_URI = NEON_URI.replace("postgresql://", "postgresql+psycopg://")
# The connection pool uses the standard prefix
DB_POOL_URI = NEON_URI

pool = ConnectionPool(
    conninfo=DB_POOL_URI, 
    min_size=1,
    max_size=5,
    kwargs={
        "autocommit": True, 
        "row_factory": dict_row,
        # Tells psycopg to send keepalive packets so the cloud host doesn't drop idle sockets as quickly
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    } 
)

#print("🧠 Connecting Spezi to the Hybrid Vector Knowledge Base...")
engine = PGEngine.from_connection_string(DB_URI)

# Uses the Hugging Face API for embeddings 
print("🧠 Connecting to Hugging Face Embeddings...")
embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-m3", huggingfacehub_api_token=os.getenv("HF_TOKEN"))

"""
engine.init_vectorstore_table(
    table_name="spezi_idiom_knowledge",
    vector_size=1024,
)
"""

vector_store = PGVectorStore.create_sync(
    engine=engine,
    table_name="spezi_idiom_knowledge",
    embedding_service=embeddings,
    hybrid_search_config=HybridSearchConfig(
        fusion_function=reciprocal_rank_fusion,
        tsv_lang="pg_catalog.german" 
    )
)

# Custom graph state
class AgentState(MessagesState):
    rag_context: str  # Safely passes retrieved DB context to synthesis without polluting message history

# Define the Tool 
# The docstring here is CRITICAL. Llama 3.2 reads this to decide WHEN to use the tool.
@tool
def search_german_idioms(query: str) -> str:
    """Searches the local database ONLY for specific German idioms or colloquial expressions.
    
    CRITICAL GUARDRAILS:
    - DO NOT call this tool for greetings (e.g., 'Hi', 'Hello', 'Guten Tag', 'How are you').
    - DO NOT use this tool for general conversations, small talk or general translations.
    - DO NOT use this tool when asked to explain German grammar.
    - DO NOT use this tool for memory checks or about the conversation history (e.g., 'do you remember me?', 'what did we talk about?').
    - Only invoke this tool when the user explicitly asks to translate, explain, or find a specific German or English idiom.
    """
    pass


# Initialize the LLM
# Instance A: Clean model without tools attached
print("🚀 Connecting to Groq Cloud LLM...")
llm_no_tool = ChatGroq(
    model="openai/gpt-oss-120b", # for multilingual and sending syntax correct prompt for tool calling
    temperature=0.3,
)

# Instance B: Decision model with tools bound
llm_with_tool = llm_no_tool.bind_tools([search_german_idioms], tool_choice="auto")


#Node FUNCTIONS

def compact_history_node(state:AgentState):
    # Prunes chat history if it grows too long, preserving context via a summary.
    messages = state["messages"]
    
    # Prune history if it exceeds 6 messages (3 turns)
    if len(messages) <= 10:
        return {"messages": []}
        
    #print(f"\n⚙️ [DB Optimization] State has {len(messages)} messages. Compacting data rows...")
    
    messages_to_summarize = messages[:-2]
    #recent_messages = messages[-2:]
    
    summary_prompt = (
        "Progressively summarize the following conversation between a human and an AI agent named Spezi. "
        "Keep the summary concise and capture user's personal information, preferences and core goals."
        "Remember the instructions given by the user."
        "Do not include any pleasantries:\n\n"
    )
    
    for msg in messages_to_summarize:
        if isinstance(msg, SystemMessage) and "Summary of earlier conversation" in msg.content:
            # Carry over old summaries instead of wrapping them repeatedly
            summary_prompt += f"PREVIOUS SUMMARY: {msg.content}\n"
        else:
            summary_prompt += f"{msg.type.upper()}: {msg.content}\n"

    summary_response = llm_no_tool.invoke(summary_prompt)
    summary_text = f"Summary of earlier conversation: {summary_response.content}"
    
    # print(f"✅ Generated Summary: '{summary_response.content[:50]}...'")
    
    # LangGraph standard: Send a RemoveMessage list to wipe old database rows permanently
    delete_old_rows = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id is not None]
    
    # Rebuild state: Drop old records and inject the summary
    return {
        "messages": delete_old_rows + [SystemMessage(content=summary_text)] 
    }


def decision_node(state: AgentState):
    """Node 1: Evaluates user input and determines if a tool call is required."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SPEZI_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])
    filled_prompt = prompt.invoke({"messages": state["messages"]})
    response = llm_with_tool.invoke(filled_prompt, tool_choice="auto")

    # Fix1:  Guarantee the message has an ID so it can be deleted later
    if not getattr(response, "id", None):
        response.id = str(uuid.uuid4())
    return {"messages": [response]}


def db_lookup_node(state: AgentState):
    """Node 2: Executes hybrid pgvector search and purges the tool call message from history."""
    last_msg = state["messages"][-1]
    
    # Extract search query generated by decision node
    tool_call = last_msg.tool_calls[0]
    query = tool_call["args"].get("query", "")
    
    print(f"\n⚙️ [Database Search] Querying pgvector for idiom: '{query}'")
    retrieved_docs = vector_store.similarity_search(query, k=2)
    
    if not retrieved_docs:
        context_str = "No matching idiom found in database. Answer using general knowledge."
    else:
        context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # Safe removal check
    removal = [RemoveMessage(id=last_msg.id)] if getattr(last_msg, "id", None) else []
        
    # Purge the tool_call AIMessage to prevent raw syntax pollution in PostgreSQL state
    return {
        "messages": removal, 
        "rag_context": context_str
    }


def synthesis_node(state: AgentState):
    """Node 3: Formulates final user response using clean LLM instance."""
    rag_context = state.get("rag_context", "")
    
    if rag_context:
        dynamic_system = (
            f"{SPEZI_SYSTEM_PROMPT}\n\n"
            f"### SYSTEM ALERT: DATABASE SEARCH COMPLETE ###\n"
            f"A background search has ALREADY been executed for the user's latest question.\n"
            f"RESULTS FOUND:\n{rag_context}\n"
            f"##############################################\n\n"
            f"CRITICAL INSTRUCTION: You MUST NOT output JSON, and you MUST NOT attempt to call any tools or search functions. "
            f"Your only job is to read the results above and answer the user directly in natural language."
        )
    else:
        dynamic_system = SPEZI_SYSTEM_PROMPT

    # FIX2: Failsafe to strip out orphaned tool calls before hitting the clean LLM
    clean_messages = [msg for msg in state["messages"] if not getattr(msg, "tool_calls", None)]
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", dynamic_system),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    filled_prompt = prompt.invoke({"messages": clean_messages})
    
    # Executed via llm_no_tool: direct answer output
    response = llm_no_tool.invoke(filled_prompt)
    
    # Clear rag_context state variable for the next turn
    return {"messages": [response], "rag_context": ""}


# Routing Condition
def route_decision(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "db_lookup"
    return END


#  Build the LangGraph State Machine
workflow = StateGraph(AgentState)

# Add operational blocks
workflow.add_node("compactor", compact_history_node)
workflow.add_node("decision", decision_node)
workflow.add_node("db_lookup", db_lookup_node)
workflow.add_node("synthesis", synthesis_node)


# FIX: Set the clean structural routing path
workflow.add_edge(START, "compactor")   # 1. Clean the database history first
workflow.add_edge("compactor", "decision") 
       
# Spezi decides: Does he need a tool, or should he just talk to the user with llm_no_tool?
workflow.add_conditional_edges("decision", route_decision)

#Linear Forward path
workflow.add_edge("db_lookup", "synthesis")
workflow.add_edge("synthesis",END)

# 4. Global Checkpointer Setup
"""
with pool.connection() as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    """

#checkpointer_conn = psycopg.connect(DB_POOL_URI, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(pool)
checkpointer.setup()

spezi_app = workflow.compile(checkpointer=checkpointer)

def run_spezi():
    print("\n 🥤 Spezi is cold, carbonated, and live! Type 'exit' to end the chat.")
    print("-" * 50)
    user_id = input("\n Enter User ID (e.g., Fanta): ").strip()
    if not user_id:
        user_id = "default_user"
        
    print(f"\n--- Chat session initialized for: {user_id} ---")
    config = {
                "configurable": {"thread_id": user_id},
            }
    
    while True:
        try:
            user_in = input(f"\n[{user_id}] You: ")
            if user_in.strip().lower() == "exit":
                print("Spezi: Tschüss! Catch you later!")
                break
                
            if not user_in.strip():
                continue

            input_data = {"messages": [("user", user_in)]}
            
            output = spezi_app.invoke(input_data, config=config)
            
            latest_reply = output["messages"][-1].content
            print(f"\nSpezi: {latest_reply}")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

    pool.close()

if __name__ == "__main__":
    run_spezi()



def ask_spezi(user_id: str, message: str) -> str:
    """Executes a single conversational turn for a given user thread."""
    config = {
        "configurable": {"thread_id": user_id},
        "recursion_limit": 6
    }
    input_data = {"messages": [("user", message)]}
    
    output = spezi_app.invoke(input_data, config=config)
    
    # Return the latest assistant response
    return output["messages"][-1].content