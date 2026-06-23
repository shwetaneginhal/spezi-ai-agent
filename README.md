# Spezi: a friendly, local German language teaching AI Agent 

The goal of Sepzi is to help an English speaker learn how to talk like a true German local.  This is built using a **local deployment topology**. Spezi is available offline with unlimited usage and optimization and control.

The core architecture is a multi-node **LangGraph state-machine structure** with a **Hybrid search Router RAG Pipeline** (Sparse Keyword + Dense Semantic Search) using a unified database. Spezi has a **persistence memory** (sliding-window compactor) for each user, allowing for long conversations that can be picked up at any time. 

---

## Tech Stack

**Agent Framework:** LangGraph, LangChain, LangChain Postgres

**Foundational Models:** Llama 3.2 & BGE-M3 (Multilingual Embedding Engine)

**Database & Vector Layer:** PostgreSQL 16 + pgvector (Dockerized)

**Database Driver System:** psycopg3 (thread-safe ConnectionPool with dynamic dict_row factories and explicit transaction autocommit mechanics)


## Architecture and Core Features

### 1. Unified Database (Postgres + `pgvector`) 
To keep the architecture simple and avoid managing multiple databases, this project consolidates all storage in a single local Postgres instance:

* Chat History: Manages active message logs and multi-turn conversation threads for each user_id.
* Vector Search: Handles semantic text search directly within Postgres using the pgvector extension.

### 2. Agent Workflow (LangGraph)
Built on LangGraph, the agent operates as a cyclic graph with processing nodes. This structure guarantees predictable state transitions and precise control over conversation history and tool calling. Instead of forcing database hits on every turn, the graph utilizes conditional routing edges.

```text
       [ User Input ]
             │
             ▼
    ┌─────────────────┐
    │  1. Compactor   │ ──► Progressively cleans & summarizes dialogue
    └─────────────────┘
             │
             ▼
    ┌─────────────────┐ ◄──────────────────────────────┐
    │  2. Spezi LLM   │                                │
    └─────────────────┘                                │
             │                                         │
             ▼                                         │ (Loop back with data)
     [tools_condition]                                 │
             │                                         │
             ├───(Yes, Idiom query) ──────► ┌───────────────────┐
             │                              │   3. ToolNode     │
             │                              └───────────────────┘
             │
             └───(No, regular chat & Spezi replies)───────► [ END ]

```

### 2. RAG pipeline with Hybrid search 

#### External Dataset

Source: https://github.com/marziehf/IdiomTranslationDS/tree/master

Filtered out the en-de subset because it maps English cultural idioms into German text. Instead, I isolated the de-en subset to capture authentic German-born Umgangssprache, chunking the parallel data into bi-directional templates to allow user's to query the RAG pipeline interchangeably in both English and German.

#### Chunking strategy

Used **bi-directional strategy** to fuse both en-de and then de-en into a single, labelled text chunk. 

Example :- 
```
Dataset: German Umgangssprache
Target Idiom (Base Form): neu maßstab setzen
English Meaning: to create a milestone
German Contextual Example: Dank unseres Engagements setzen wir immer wieder neue Maßstäbe in der Automatisierungs- und Antriebstechnik.
English Translation: Thanks to our commitment , we continue to set new standards in automation and drive technology.
```

#### Embedding Model - BGE-M3 (a multilingual model)

BAAI **BGE-M3** is the open-source model for multilingual tasks. It is trained on over 100 languages and natively understands German nuances. It is a Hybrid Retrieval model. In a single forward pass, it can generate Dense vectors (for semantic meaning), Sparse vectors (for keyword matching) and multi-vector (for analyzing token-token interaction) outputs. 

In **Hybrid search**, because keyword matches and semantic meaning use completely different scoring methods, their results cannot be compared directly.  To fix this, **Reciprocal Rank Fusion (RRF)** is used here. RRF looks at how high a document ranks in both lists, safely merges and normalizes these results into a single, highly accurate list.

## 🗺️ Roadmap

- [x] Prompt Engineering
- [x] Adding Persistent Memory (context retention)
- [x] Implementing Router RAG with Tool calling
- [ ] **[Upcoming]** Evaluation
- [ ] **[Upcoming]** API Deployment and UI











