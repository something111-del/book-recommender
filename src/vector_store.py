import chromadb
from chromadb.utils import embedding_functions
from .config import DATA_DIR, OPENAI_API_KEY, EMBEDDING_MODEL
import pandas as pd
import os

# Initialize Chroma Client
# PersistentClient saves to disk
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

# Use OpenAI embedding function for Chroma
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL
)

def get_or_create_collection(name="books"):
    return client.get_or_create_collection(name=name, embedding_function=openai_ef)

def add_books_to_vector_store(df, collection_name="books"):
    """
    Adds books to the ChromaDB collection.
    Expects df to have 'description', 'title', 'isbn13', 'categories', etc.
    """
    collection = get_or_create_collection(collection_name)
    
    # Prepare data
    documents = df['description'].tolist()
    metadatas = df.apply(lambda x: {
        "title": x['title'],
        "isbn13": str(x['isbn13']),
        "categories": str(x['categories']),
        "predicted_category": str(x.get('predicted_category', 'Unknown')),
        "authors": str(x['authors']),
        "thumbnail": str(x['thumbnail'])
    }, axis=1).tolist()
    ids = [str(x) for x in df['isbn13'].tolist()]
    
    # Add to collection (Chroma handles batching usually, but let's be safe if huge)
    # Upsert to avoid duplicates
    print(f"Adding {len(documents)} documents to vector store...")
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("Books added to vector store.")

def query_books(query_text, n_results=5, collection_name="books", filter_category=None):
    """
    Queries the vector store.
    """
    collection = get_or_create_collection(collection_name)
    
    where_filter = None
    if filter_category and filter_category != "All":
        where_filter = {"predicted_category": filter_category}
        
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter
    )
    
    return results
