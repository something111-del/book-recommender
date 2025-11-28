import pandas as pd
from .config import PROCESSED_DATA_DIR
from .vector_store import add_books_to_vector_store
import sys

def build_db():
    processed_file = PROCESSED_DATA_DIR / "books_processed.csv"
    if not processed_file.exists():
        print(f"Error: {processed_file} not found. Run preprocess.py first.")
        sys.exit(1)
        
    print(f"Loading processed data from {processed_file}...")
    df = pd.read_csv(processed_file)
    
    # Check if empty
    if df.empty:
        print("Dataframe is empty.")
        sys.exit(1)
        
    print("Building vector store...")
    add_books_to_vector_store(df)
    print("Vector store built successfully.")

if __name__ == "__main__":
    build_db()
