"""
Initialize ChromaDB vector database with OpenAI embeddings.
This script creates a persistent vector database from the tagged book descriptions.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os

def initialize_vector_db():
    print("Loading environment variables...")
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    print("Loading tagged descriptions...")
    raw_documents = TextLoader("tagged_description.txt").load()
    documents = [Document(page_content=line) for line in raw_documents[0].page_content.split("\n") if line.strip()]
    
    print(f"Creating vector database with {len(documents)} documents...")
    print("This will take several minutes as it generates embeddings using OpenAI...")
    
    # Create persistent vector database
    db = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory="./data/chroma_db"
    )
    
    print("Vector database initialized successfully!")
    print(f"Database saved to: ./data/chroma_db")
    print(f"Total documents: {len(documents)}")

if __name__ == "__main__":
    initialize_vector_db()
