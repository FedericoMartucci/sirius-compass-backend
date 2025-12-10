import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings_model():
    """
    Returns the embedding model configured for the project.
    We use Google's 'embedding-001' model which works well with Gemini.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY for embeddings.")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )