# Spezi: a friendly, local German language teaching AI Agent 

Spezi is an interactive AI language tutor designed to help users learn German idioms, local expressions and conversational nuance. Unlike standard AI chatbots that suffer from unpredictable behavior or "thinking" loops, Spezi is built on a **single-pass deterministic architecture**. It intelligently routes user requests, actively manages long-term conversation memory to prevent context bloat, and retrieves authentic German idioms from a hybrid database.

---

## 🌐 Live Demo & Instant Access

* Interactive Frontend: Live Streamlit Web Application (No setup required)
    Link: https://spezi-ai-agent.streamlit.app/

* **Architecture Overview:** The application runs on a decoupled cloud infrastructure. The Streamlit frontend communicates directly via HTTPS with a containerized FastAPI backend hosted on Render. User session persistence and vector search are powered by a serverless Neon PostgreSQL database.

Note on Free-Tier Hosting: The backend is hosted on Render's free tier. If the application has been inactive for more than 15 minutes, the initial request may take 30–45 seconds to perform a cold boot. Subsequent responses will be immediate.

---

## Project Evolution: From Version 1 (agent_v1) to Production Architecture

The initial version of this project (agent_v1.py) was built on Llama 3.2 as a local, offline agent with **ReAct pattern** (cyclic, tool-binding loop `spezi` ↔ `tools`). While it returned results for queries, this early version had two major **limitations**: 
1. **Unnecessary Database Calls:** It executed idioms database searches on every turn (even for general conversations like greetings).
2. **Loop Risks & Memory Constraints:** The local LLM occasionally got stuck in repetitive thinking loops during tool selection, and local GPU memory constrained the model size.

The architecture was completely redesigned into a **single-pass execution graph** driven by an explicit **Semantic Router node** and this solved the repetitive loop. As general queries were still hitting the database and because of the local GPU memory constraints, it was moved into an online agent with bigger Llama model - llama-3.3-70b (via Groq API). This ensured accurate intent classification and predicatble routing latency.

---

## 🏗️ Architecture Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'lineColor': '#555'}}}%%
graph TD
    %% Define Color Classes
    classDef client fill:#e0f7fa,stroke:#006064,stroke-width:2px,color:#000;
    classDef server fill:#b3e5fc,stroke:#01579b,stroke-width:2px,color:#000;
    classDef graphNode fill:#e1bee7,stroke:#4a148c,stroke-width:2px,color:#000;
    classDef startEnd fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000;
    classDef externalAPI fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000;
    classDef database fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000;

    User([User Browser]):::client <-->|HTTPS| Frontend[Streamlit Frontend<br/>Streamlit Community Cloud]:::client
    Frontend <-->|HTTP POST /chat| Backend[FastAPI Microservice<br/>Hosted on Render]:::server

    subgraph LangGraph Execution Engine
        Backend --> START([START]):::startEnd
        START --> Compactor[1. Compactor Node<br/>History Summarization & State Pruning]:::graphNode
        Compactor --> Decision[2. Decision Node<br/>Intent Routing & Tool Selection]:::graphNode
        
        Decision -->|Tool Call Required| DBLookup[3. DB Lookup Node<br/>Hybrid RAG]:::graphNode
        Decision -->|Direct Response| END([END]):::startEnd
        
        DBLookup --> Synthesis[4. Synthesis Node<br/>Contextual Response Generation]:::graphNode
        Synthesis --> END
    end

    subgraph External Infrastructure & APIs
        Decision <-->|LLM Inference| Groq[Groq Cloud API<br/>llama-3.3-70b-versatile]:::externalAPI
        Synthesis <-->|LLM Inference| Groq
        DBLookup <-->|Vector Embeddings| HF[Hugging Face Serverless API<br/>BAAI/bge-m3]:::externalAPI
        DBLookup <-->|Hybrid Search| NeonDB[(Neon Serverless PostgreSQL<br/>pgvector + Full-Text Search)]:::database
        Compactor <-->|Thread Checkpoints| NeonDB
    end
```

## 🛠️ Tech Stack

| Component | Tools | Purpose |
| :--- | :--- | :--- |
| **Language & Web** | Python 3.10+, FastAPI, Uvicorn | REST Backend Service |
| **Frontend** | Streamlit | Web chat Interface |
| **Agent Orchestration** | LangGraph, LangChain Core | State machine workflow and thread memory |
| **LLM Inference** | Groq API (`llama-3.3-70b-versatile`) | Intent evaluation & text synthesis |
| **Embeddings** | Hugging Face API (`BAAI/bge-m3`) | Serverless 1024-dim dense vector generation |
| **Database** | Neon Serverless PostgreSQL (`pgvector`) | Vector store & thread checkpointer |
| **Hosting** | Render (Backend), Streamlit Cloud (Frontend) | Live Production deployment |

---

## Key Engineering & Architectural Highlights


### 1. Deterministic State Workflow 
Rather than relying on an unconstrained cyclic agent loop, the orchestrator executes a strict, multi-stage state graph (agent_core.py):

* **Compactor Node (Memory management):** Evaluates session memory before every turn. If the conversation gets too long, it progressively summarizes older messages and explicitly deletes obsolete rows from the PostgreSQL database to prevent context bloat.

* **Decision Node (Semantic Routing):** Evaluates user intent first. If the user just says "Hello," it skips the database entirely, saving compute time and API costs.

* **DB Lookup Node (State Hygiene):** When a tool call is generated, this node intercepts the invocation, executes a hybrid vector search against PostgreSQL, and immediately purges the raw JSON tool_calls message from history. This eliminates syntax artifacts.

* **Synthesis Node:** Uses an un-tooled LLM instance to formulate final natural conversational responses by combining user input with the cleanly injected rag_context (if invoked). 

### 2.Hybrid RAG Pipeline (Semantic + Lexical Search)
Knowledge retrieval relies on a dual-strategy lookup against a Neon PostgreSQL database:

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

## 📂 Project structure

```text
.
├── agent_core.py       # Core LangGraph state machine, nodes, and checkpointer logic
├── server.py           # FastAPI application exposing REST endpoints
├── frontend.py         # Streamlit UI application
├── prompts.py          # System prompt definition and guardrail specifications
├── extractIdioms.py    # Raw dataset extraction script generating semantic_chunks.jsonl
├── ingest_idioms.py    # Batch ingestion script populating pgvector
├── requirements.txt    # Project dependencies
└── README.md           # Documentation

```

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
---

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

