# agent_core.py
import psycopg
from psycopg.rows import dict_row  # Crucial for dictionary row mapping
from psycopg_pool import ConnectionPool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from prompts import SPEZI_SYSTEM_PROMPT
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_postgres import PGEngine, PGVectorStore
try:
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
except ModuleNotFoundError:
    try:
        from langchain_postgres.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
    except ModuleNotFoundError:
        from langchain_postgres import HybridSearchConfig, reciprocal_rank_fusion

# DB Connection Configuration (Fixed with production pool properties)
DB_POOL_URI = "postgresql://spezi_admin:spezi_pass@localhost:5432/spezi_db"

pool = ConnectionPool(
    conninfo=DB_POOL_URI, 
    max_size=10,
    kwargs={"autocommit": True, "row_factory": dict_row} # Crucial fix for LangGraph checkpointers
)


#print("🧠 Connecting Spezi to the Hybrid Vector Knowledge Base...")

DB_URI = "postgresql+psycopg://spezi_admin:spezi_pass@localhost:5432/spezi_db"
engine = PGEngine.from_connection_string(DB_URI)
embeddings = OllamaEmbeddings(model="bge-m3")

vector_store = PGVectorStore.create_sync(
    engine=engine,
    table_name="spezi_idiom_knowledge",
    embedding_service=embeddings,
    hybrid_search_config=HybridSearchConfig(
        fusion_function=reciprocal_rank_fusion,
        tsv_lang="pg_catalog.german" 
    )
)

# 2. Define the Tool 
# The docstring here is CRITICAL. Llama 3.2 reads this to decide WHEN to use the tool.
@tool
def search_german_idioms(query: str) -> str:
    """Searches the local database ONLY for specific German idioms, colloquial slang phrases, or cultural expressions.
    
    CRITICAL GUARDRAILS:
    - DO NOT use this tool for general chat, general translations.
    - DO NOT use this tool when asked to explain German grammar.
    - DO NOT use this tool for meta-questions about the conversation history (e.g., 'do you remember me?', 'what did we talk about?').
    - Only invoke this tool when the user explicitly asks to translate, explain, or find a German or English idiom.
    """
    
    #print(f"\n⚙️ [Agent Decision] Spezi decided to search DB for: '{query}'")
    retrieved_docs = vector_store.similarity_search(query, k=2)
    #print(retrieved_docs)
    
    if not retrieved_docs:
        #print("\n No matches retreived")
        return "No matching idiom found in database."
    
        
    return "\n---\n".join([doc.page_content for doc in retrieved_docs])


# Initialize the LLM
llm = ChatOllama(
    model="llama3.2", 
    temperature=0.1,
    validate_model_on_init=True,
    num_gpu=-1
)

# Bind the tool to the LLM so it knows it exists
tools = [search_german_idioms]
llm_with_tools = llm.bind_tools(tools)

# We inject {rag_context} so Spezi can read the database results
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SPEZI_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

'''
class TutorState(MessagesState):
    rag_context: str
    '''


# Node 1: The Auto-Compactor (Database Pruning Node)
def compact_history_node(state: MessagesState):
    messages = state["messages"]
    
    # Prune history if it exceeds 6 messages (3 turns)
    if len(messages) <= 10:
        return {"messages": []}
        
    #print(f"\n⚙️ [DB Optimization] State has {len(messages)} messages. Compacting data rows...")
    
    messages_to_summarize = messages[:-2]
    #recent_messages = messages[-2:]
    
    summary_prompt = (
        "Progressively summarize the following conversation between a human and an AI agent named Spezi. "
        "Keep the summary concise and capture user's personal information, slang and core goals."
        "Remember the instructions given by the user."
        "Do not include any pleasantries:\n\n"
    )
    
    for msg in messages_to_summarize:
        if isinstance(msg, SystemMessage) and "Summary of earlier conversation" in msg.content:
            # Carry over old summaries instead of wrapping them repeatedly
            summary_prompt += f"PREVIOUS SUMMARY: {msg.content}\n"
        else:
            summary_prompt += f"{msg.type.upper()}: {msg.content}\n"

    summary_response = llm.invoke(summary_prompt)
    summary_text = f"Summary of earlier conversation: {summary_response.content}"
    
    # print(f"✅ Generated Summary: '{summary_response.content[:50]}...'")
    
    # LangGraph standard: Send a RemoveMessage list to wipe old database rows permanently
    delete_old_rows = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id is not None]
    
    # Rebuild state: Drop old records and inject the summary
    return {
        "messages": delete_old_rows + [SystemMessage(content=summary_text)] 
    }

'''
# Node 2 - Hybrid Retriever
def hybrid_retrieval_node(state: MessagesState):
    # Grab the user's most recent message
    latest_user_message = state["messages"][-1].content
    print(f"\n🔍 [Hybrid Search] Querying pgvector for: '{latest_user_message}'...")
    
    # Execute the Hybrid Search (BGE-M3 Semantic + Postgres Lexical)
    retrieved_docs = vector_store.similarity_search(latest_user_message, k=2)
    
    if not retrieved_docs:
        return {"rag_context": "No matching idiom found. Rely on your general knowledge."}
        
    # Compile the retrieved documents into a clean string for Spezi to read
    context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    print(f"✅ Retrieved {len(retrieved_docs)} matching idiom reference(s)!")
    
    # Save the string into our Custom State
    return {"rag_context": context_str}
'''

# Node 3: Execute chat logic
def call_spezi_model(state: MessagesState):
    # Safely get the context (defaults to empty if the retriever found nothing)
    #rag_context = state.get("rag_context", "No idiom reference available.")
    
    filled_prompt = prompt_template.invoke({
        "messages": state["messages"],
       # "rag_context": rag_context
    })
    response = llm_with_tools.invoke(filled_prompt)
    return {"messages": [response]}


# 3. Build the LangGraph State Machine
workflow = StateGraph(MessagesState)

# Add operational blocks
workflow.add_node("compactor", compact_history_node)
#workflow.add_node("retriever", hybrid_retrieval_node)
workflow.add_node("spezi", call_spezi_model)
workflow.add_node("tools", ToolNode(tools))

# FIX: Set the clean structural routing path
workflow.add_edge(START, "compactor")   # 1. Clean the database history first
workflow.add_edge("compactor", "spezi") # 2. call LLM
       
# Spezi decides: Does he need a tool, or should he just talk to the user?
workflow.add_conditional_edges("spezi", tools_condition)

# If he used a tool, feed the database results back to his brain to generate the final answer
workflow.add_edge("tools", "spezi")

# 4. Global Checkpointer Setup
with pool.connection() as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

spezi_app = workflow.compile(checkpointer=checkpointer)

def run_spezi():
    print("\n 🥤 Spezi is cold, carbonated, and live! Type 'exit' to end the chat.")
    print("-" * 50)
    user_id = input("\n Enter User ID (e.g., test_user): ").strip()
    if not user_id:
        user_id = "default_user"
        
    print(f"\n--- Chat session initialized for: {user_id} ---")
    config = {"configurable": {"thread_id": user_id}}
    
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