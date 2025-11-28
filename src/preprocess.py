import pandas as pd
import numpy as np
from transformers import pipeline
from .config import PROCESSED_DATA_DIR, ZERO_SHOT_MODEL
from .data_loader import load_raw_data
import torch

def clean_data(df):
    """
    Basic data cleaning.
    """
    print("Cleaning data...")
    # Drop rows with missing essential fields
    df = df.dropna(subset=['description', 'title'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['isbn13'], keep='first')
    
    # Filter out very short descriptions
    df = df[df['description'].str.len() > 20]
    
    return df

def classify_books(df, sample_size=None):
    """
    Uses Zero-Shot Classification to assign genres/categories.
    If sample_size is provided, only processes a subset (for testing/speed).
    """
    print(f"Initializing Zero-Shot Classifier with {ZERO_SHOT_MODEL}...")
    device = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
    classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=device)
    
    candidate_labels = [
        "Fiction", "Science Fiction", "Fantasy", "Mystery", "Thriller", 
        "Romance", "History", "Biography", "Self-Help", "Business", 
        "Science", "Technology", "Programming", "Philosophy", "Psychology"
    ]
    
    if sample_size:
        print(f"Processing a sample of {sample_size} books...")
        df = df.head(sample_size).copy()
    
    print("Classifying books...")
    
    # Batch processing could be faster, but simple loop for now or apply
    # Using a simple loop with progress print for visibility
    
    predictions = []
    descriptions = df['description'].tolist()
    
    # Process in batches to avoid memory issues if large
    batch_size = 16
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        results = classifier(batch, candidate_labels)
        for res in results:
            predictions.append(res['labels'][0]) # Take the top label
        
        if i % 32 == 0:
            print(f"Processed {i}/{len(descriptions)}...")
            
    df['predicted_category'] = predictions
    return df

def run_pipeline(sample_size=None):
    df = load_raw_data()
    df = clean_data(df)
    
    # For the sake of the user's request, we will classify. 
    # Note: Running this on 7k books might take a while on CPU/MPS.
    # We might want to warn the user or just do it.
    df = classify_books(df, sample_size=sample_size)
    
    output_path = PROCESSED_DATA_DIR / "books_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Run on a small sample by default for testing, or remove arg for full run
    # The user wants to build the system, likely wants full data but let's be careful with time.
    # I'll set a default sample size of 50 for the 'main' run to verify it works quickly.
    run_pipeline(sample_size=50)
