# Spezi: a friendly, local German language teaching AI Agent 

The goal of Sepzi is to help an English speaker learn how to talk like a German local. Spezi is a containerized, cloud-native REST backend and interactive frontend for conversational German language learning. It is designed to imitate a 26-year-old local living in Berlin. 

The system uses a **Semantic Router RAG** architecture, implemented as a Directed Acyclic Graph (DAG) in **LangGraph** to ensure deterministic execution and prevent runaway agentic loops. Spezi has a unified database (PostgreSQL + pgvector) with **persistence memory** (sliding-window compactor) for each user, allowing for long conversations that can be picked up at any time. 

---

## 🌐 Live Demo & Instant Access
Spezi is fully deployed and publicly accessible online. You can interact with the live AI tutor immediately without any local installation by visiting the Streamlit Web Interface - https://spezi-ai-agent.streamlit.app/

The application runs on a decoupled cloud architecture where the Streamlit frontend communicates directly over HTTPS with the containerized FastAPI backend hosted on Render, this handles session persistence via a serverless Neon PostgreSQL vector database.

Note on Performance: The backend is hosted on Render's free tier. If the application has been inactive for more than 15 minutes, the initial request may take 30–45 seconds to perform a cold boot. Subsequent responses will be immediate.

---

## 🏗️ Architecture Diagram

```mermaid
graph TD
    User([User Browser]) <-->|HTTPS| Frontend[Streamlit Frontend<br/>Streamlit Community Cloud]
    Frontend <-->|HTTP POST /chat| Backend[FastAPI Microservice<br/>Hosted on Render]

    subgraph LangGraph Execution Engine
        Backend --> START([START])
        START --> Compactor[1. Compactor Node<br/>History Summarization & State Pruning]
        Compactor --> Decision[2. Decision Node<br/>Intent Routing & Tool Selection]
        
        Decision -->|Tool Call Required| DBLookup[3. DB Lookup Node<br/>Hybrid RAG]
        Decision -->|Direct Response| END([END])
        
        DBLookup --> Synthesis[4. Synthesis Node<br/>Contextual Response Generation]
        Synthesis --> END
    end

    subgraph External Infrastructure & APIs
        Decision <-->|LLM Inference| Groq[Groq Cloud API<br/>llama-3.3-70b-versatile]
        Synthesis <-->|LLM Inference| Groq
        DBLookup <-->|Vector Embeddings| HF[Hugging Face Serverless API<br/>BAAI/bge-m3]
        DBLookup <-->|Hybrid Search| NeonDB[(Neon Serverless PostgreSQL<br/>pgvector + Full-Text Search)]
        Compactor <-->|Thread Checkpoints| NeonDB
    end
```

## Key Engineering & Architectural Highlights


### 1. Deterministic State Workflow (DAG)
Rather than relying on an unconstrained cyclic agent loop, the orchestrator executes a strict, multi-stage state graph (agent_core.py):

* **Compactor Node:** Evaluates session memory before invocation. If conversation length exceeds thresholds, it progressively summarizes older turns into a concise SystemMessage summary and issues RemoveMessage commands to wipe obsolete rows from PostgreSQL.

* **Decision Node:** Evaluates user intent using llama-3.3-70b-versatile with the database tool schema.

* **DB Lookup Node:** When a tool call is generated, this node intercepts the invocation, executes a hybrid vector search against PostgreSQL, and immediately purges the raw JSON tool_calls message from history using RemoveMessage. This eliminates syntax artifacts.

* **Synthesis Node:** Uses an un-tooled LLM instance to formulate natural conversational responses by combining user input with the cleanly injected rag_context (if invoked).

### 2.Hybrid RAG Pipeline (Semantic + Lexical)
Knowledge retrieval relies on a dual-strategy lookup against a Neon PostgreSQL database using langchain_postgres:

* **Dense Semantic Search (Vector Search):** Generates 1024-dimensional embeddings via Hugging Face’s Serverless API using BAAI/bge-m3.

* **Sparse Lexical Search (Keyword search):** Utilizes PostgreSQL tsvector with German dictionary parsing (pg_catalog.german) for exact lexical matching.

* Fusion: Merges semantic and keyword results using Reciprocal Rank Fusion (RRF).

###  3.Bi-Directional Dataset Parsing & Ingestion
External dataset [Source] : https://github.com/marziehf/IdiomTranslationDS/tree/master

* **Structured Parsing** *(extractIdioms.py)***:** Isolated the de-en subset to capture authentic German Umgangssprache, chunking the parallel data into bi-directional templates to allow users to query the RAG pipeline interchangeably in both English and German.

Example :- 
```
Dataset: German Umgangssprache
Target Idiom (Base Form): neu maßstab setzen
English Meaning: to create a milestone
German Contextual Example: Dank unseres Engagements setzen wir immer wieder neue Maßstäbe in der Automatisierungs- und Antriebstechnik.
English Translation: Thanks to our commitment , we continue to set new standards in automation and drive technology.
```

* **Batch Ingestion** *(ingest_idioms.py)***:** Loads prepared documents into the PostgreSQL spezi_idiom_knowledge table, provisioning dense vector indices and full-text search dictionaries.

###  4.Decoupled Web Tier
* **Backend** *(server.py)***:** Built with FastAPI and Uvicorn, serving asynchronous */chat* and */health* REST endpoints.

* **Frontend** *(frontend.py)***:** Lightweight Streamlit UI handling user sessions, chat rendering, and API communication.

---

## 🛠️ Tech Stack

| Component | Tools | Purpose |
| :--- | :--- | :--- |
| **Language & Web** | Python 3.10+, FastAPI, Uvicorn | REST Backend Service |
| **Frontend** | Streamlit | Web Interface |
| **Agent Orchestration** | LangGraph, LangChain Core | State machine workflow and thread memory |
| **LLM Inference** | Groq API (`llama-3.3-70b-versatile`) | Intent evaluation & text synthesis |
| **Embeddings** | Hugging Face API (`BAAI/bge-m3`) | Serverless 1024-dim dense vector generation |
| **Database** | Neon Serverless PostgreSQL (`pgvector`) | Vector store & thread checkpointer |
| **Hosting** | Render (Backend), Streamlit Cloud (Frontend) | Production deployment |


## 📂 Repository structure
.
├── agent_core.py       # Core LangGraph state machine, nodes, and checkpointer logic
├── server.py           # FastAPI application exposing REST endpoints
├── frontend.py         # Streamlit UI application
├── prompts.py          # System prompt definition and guardrail specifications
├── extractIdioms.py    # Raw dataset extraction script generating semantic_chunks.jsonl
├── ingest_idioms.py    # Batch ingestion script populating pgvector
├── requirements.txt    # Project dependencies
└── README.md           # Documentation

---

## 🚀 Environment & Setup

**For local deployment**

* Create a .env file in the project root

```bash
NEON_DB_URI=postgresql://user:password@ep-cool-endpoint.aws.neon.tech/neondb?sslmode=require
GROQ_API_KEY=gsk_your_groq_api_key
HF_TOKEN=hf_your_huggingface_access_token

```

* Clone the repo and setup requirements.txt

* Injest knowledge base

```bash
python extractIdioms.py
python ingest_idioms.py
```

* Launch FastAPI backend
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
* Launch streamlit frontend

```bash
streamlit run frontend.py
```


## 🗺️ Roadmap

- [x] Prompt Engineering 
- [x] Adding Persistent Memory (context retention)
- [x] Data ingestion and RAG implementation
- [x] Agentic workflow with tool calling using LangGraph
- [x] Deployment (FastAPI and Streamlit)
- [x] Initial evaluation and testing
- [x] Live Hosting 
- [ ] **[Upcoming]** RAG Evaluation - implement LangSmith
- [ ] **[Upcoming]** Address feedback and refine











