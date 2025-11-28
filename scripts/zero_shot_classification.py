"""
Zero-Shot Classification using facebook/bart-large-mnli

This script demonstrates how to use the BART-large-MNLI model for zero-shot
text classification to categorize books into Fiction and Nonfiction.
"""

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def load_data(filepath="data/raw/books.csv"):
    """Load the books dataset."""
    print(f"Loading data from {filepath}...")
    books = pd.read_csv(filepath)
    return books

def create_zero_shot_classifier(device="cpu"):
    """
    Create a zero-shot classification pipeline using facebook/bart-large-mnli.
    
    Args:
        device: Device to run the model on ('cpu', 'cuda', or 'mps')
    
    Returns:
        Zero-shot classification pipeline
    """
    print("Loading facebook/bart-large-mnli model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    return classifier

def classify_books(books_df, classifier, candidate_labels=["Fiction", "Nonfiction"], 
                   sample_size=None):
    """
    Classify books using zero-shot classification.
    
    Args:
        books_df: DataFrame containing book data
        classifier: Zero-shot classification pipeline
        candidate_labels: List of labels to classify into
        sample_size: Number of books to classify (None for all)
    
    Returns:
        DataFrame with classification results
    """
    # Use sample if specified
    if sample_size:
        books_to_classify = books_df.head(sample_size)
    else:
        books_to_classify = books_df
    
    print(f"Classifying {len(books_to_classify)} books...")
    
    results = []
    for idx, row in tqdm(books_to_classify.iterrows(), total=len(books_to_classify)):
        # Use description for classification
        text = row['description']
        
        # Perform zero-shot classification
        result = classifier(text, candidate_labels)
        
        # Store results
        results.append({
            'isbn13': row['isbn13'],
            'title': row['title'],
            'original_category': row.get('categories', 'Unknown'),
            'predicted_label': result['labels'][0],
            'confidence': result['scores'][0],
            'fiction_score': result['scores'][result['labels'].index('Fiction')],
            'nonfiction_score': result['scores'][result['labels'].index('Nonfiction')]
        })
    
    return pd.DataFrame(results)

def main():
    """Main execution function."""
    # Load data
    books = load_data()
    
    # Create classifier
    # Change device to 'cuda' for GPU or 'mps' for Apple Silicon
    classifier = create_zero_shot_classifier(device="cpu")
    
    # Classify a sample of books (use sample_size=None for all books)
    results = classify_books(books, classifier, sample_size=10)
    
    # Display results
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    print(results[['title', 'original_category', 'predicted_label', 'confidence']])
    
    # Save results
    output_file = "zero_shot_results.csv"
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Show accuracy if original categories are available
    if 'original_category' in results.columns:
        # Simple accuracy check (this is approximate as original categories may not match exactly)
        print(f"\nSample predictions:")
        for _, row in results.head(5).iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"  Original: {row['original_category']}")
            print(f"  Predicted: {row['predicted_label']} (confidence: {row['confidence']:.2%})")

if __name__ == "__main__":
    main()
