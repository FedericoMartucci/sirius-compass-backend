import os
import logging
from functools import lru_cache
from langchain_pinecone import PineconeVectorStore
from app.core.rag.embeddings import get_embeddings_model

logger = logging.getLogger(__name__)

def get_vector_store():
    """
    Initializes and returns the connection to the Pinecone Vector Store.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("❌ PINECONE_INDEX_NAME not found in .env")

    embeddings = get_embeddings_model()
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    return vector_store

def add_documents(texts: list[str], metadatas: list[dict] = None):
    """
    Helper to ingest text rules into the database.
    """
    store = get_vector_store()
    store.add_texts(texts=texts, metadatas=metadatas)
    logger.info(f"✅ Successfully added {len(texts)} documents to Pinecone.")

@lru_cache(maxsize=10)
def search_similar_rules(query: str, k: int = 3) -> str:
    """
    Searches for relevant engineering rules.
    
    OPTIMIZATION: Since the query for engineering standards is static 
    (we always ask for 'security, git, testing...'), we cache the result 
    in memory to avoid hitting Google and Pinecone APIs on every request.
    """
    logger.info(f"📚 Retrieving standards for query: '{query}' (Cache Miss if seen)")
    
    try:
        store = get_vector_store()
        results = store.similarity_search(query, k=k)
        
        context = "\n".join([f"- {doc.page_content}" for doc in results])
        return context
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {e}")
        return "Error retrieving standards."