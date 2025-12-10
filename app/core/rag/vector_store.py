import os
from langchain_pinecone import PineconeVectorStore
from app.core.rag.embeddings import get_embeddings_model

def get_vector_store():
    """
    Initializes and returns the connection to the Pinecone Vector Store.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in .env")

    embeddings = get_embeddings_model()
    
    # Connect to the existing index
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
    print(f"Successfully added {len(texts)} documents to Pinecone.")

def search_similar_rules(query: str, k: int = 3) -> str:
    """
    Searches for the most relevant engineering rules based on the query.
    Returns a formatted string context.
    """
    store = get_vector_store()
    results = store.similarity_search(query, k=k)
    
    context = "\n".join([f"- {doc.page_content}" for doc in results])
    return context