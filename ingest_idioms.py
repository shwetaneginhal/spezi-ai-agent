# ingest_idioms.py
import json
import os
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

# Attempt to load from the modern internal routing path
try:
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
except ModuleNotFoundError:
    try:
        from langchain_postgres.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
    except ModuleNotFoundError:
        # Fallback for alternative version structures
        from langchain_postgres import HybridSearchConfig, reciprocal_rank_fusion


# 1. Database Configuration Configurations
CONNECTION_STRING = "postgresql+psycopg://spezi_admin:spezi_pass@localhost:5432/spezi_db"
TABLE_NAME = "spezi_idiom_knowledge"
DATASET_PATH = "semantic_chunks.jsonl"

def run_hybrid_ingestion():
    # Verify dataset exists before starting up heavy models
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"❌ Could not find '{DATASET_PATH}' in the current working directory. "
            "Please ensure the file is placed next to this script."
        )

    print(f"📖 Reading structural bi-directional data from '{DATASET_PATH}'...")
    processed_docs = []
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            try:
                row_data = json.loads(line)
                chunk_content = row_data.get("semantic_chunk", "")
                chunk_metadata = row_data.get("metadata", {})
                
                # Ensure a fallback trace indicator is present in metadata
                if "line" not in chunk_metadata:
                    chunk_metadata["line"] = idx
                chunk_metadata["source"] = "idioms_de_en_master"

                # Wrap into a formal LangChain Document structure
                doc = Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                )
                processed_docs.append(doc)
            except json.JSONDecodeError as je:
                print(f"⚠️ Warning: Skipping malformed JSON row at index {idx}: {je}")
                continue

    print(f"✅ \nSuccessfully prepared {len(processed_docs)} documents for processing.")

    print("🧠\n Initializing local BGE-M3 Multilingual Embedding pipeline via Ollama...")
    # BGE-M3 naturally produces 1024-dimensional dense vectors
    embeddings = OllamaEmbeddings(model="bge-m3")

    print("🔌 Opening connection engine to PostgreSQL backend...")
    engine = PGEngine.from_connection_string(CONNECTION_STRING)

    # BGE-M3 produces dense vectors with exactly 1024 dimensions
    DENSE_VECTOR_DIMENSION = 1024

    print("🏗️\n Initializing vector table structure schema...")
    # This explicit step tells Postgres to build the table structure 
    # so that create_sync can find the required columns!
    engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=DENSE_VECTOR_DIMENSION
    )


    print("⚡\n Provisioning Hybrid Index (Vector Space + Full-Text-Search tsvector)...")
    # create_sync checks if the table exists. If it does not, it handles the execution schema matrix creation,
    # registering the text column, pgvector extension dimensions, and text search dictionaries.
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embeddings,
        # Implementing your explicit hybrid strategy configuration
        hybrid_search_config=HybridSearchConfig(
            fusion_function=reciprocal_rank_fusion,
            tsv_lang="pg_catalog.german"  # Optimizes lexical parsing dictionaries for German word variations
        )
    )

    print(f"💾 Loading vectors and text tokens to table '{TABLE_NAME}' in PostgreSQL...")
    # This chunking transfer executes the full vector math and commits them to the database row-by-row
    vector_store.add_documents(processed_docs)
    
    print("\n🎉 Success! The idiom knowledge base is completely ingested and index generation is finalized.")
    print("Spezi is now fully equipped to query German Umgangssprache idioms with Hybrid RAG lookup!")

if __name__ == "__main__":
    run_hybrid_ingestion()