import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import os

def prepare_data():
    print("Loading data...")
    # Load the books with categories
    if os.path.exists("outputs/books_with_categories.csv"):
        books = pd.read_csv("outputs/books_with_categories.csv")
    else:
        print("Error: outputs/books_with_categories.csv not found.")
        return

    print("Initializing emotion classifier (this may take a moment)...")
    # Initialize classifier
    classifier = pipeline("text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        top_k=None)
                        # device="mps") # Use mps if on Mac M1/M2/M3, otherwise remove or use "cuda" for GPU

    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    emotion_scores = {label: [] for label in emotion_labels}
    
    print(f"Processing {len(books)} books for emotion analysis...")
    print("This will take some time (approx 10-15 mins)...")

    def calculate_max_emotion_scores(predictions):
        per_emotion_scores = {label: [] for label in emotion_labels}
        for prediction in predictions:
            sorted_predictions = sorted(prediction, key=lambda x: x["label"])
            for index, label in enumerate(emotion_labels):
                per_emotion_scores[label].append(sorted_predictions[index]["score"])
        return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

    # Process books
    # Using a smaller batch for testing if needed, but running full here
    for i in tqdm(range(len(books))):
        description = str(books["description"][i])
        if len(description) > 0:
            sentences = description.split(".")
            # Filter empty sentences
            sentences = [s for s in sentences if len(s.strip()) > 0]
            if not sentences:
                sentences = [description]
                
            try:
                # Truncate to avoid max length errors if any sentence is too long
                predictions = classifier(sentences, truncation=True, max_length=512)
                max_scores = calculate_max_emotion_scores(predictions)
                for label in emotion_labels:
                    emotion_scores[label].append(max_scores[label])
            except Exception as e:
                print(f"Error processing book {i}: {e}")
                for label in emotion_labels:
                    emotion_scores[label].append(0.0)
        else:
            for label in emotion_labels:
                emotion_scores[label].append(0.0)

    # Add emotion scores to dataframe
    for label in emotion_labels:
        books[label] = emotion_scores[label]

    print("Saving books_with_emotions.csv...")
    books.to_csv("books_with_emotions.csv", index=False)

    print("Generating tagged_description.txt...")
    # Create tagged description column
    books["tagged_description"] = books["isbn13"].astype(str) + " " + books["description"].astype(str)
    
    # Save tagged description
    books["tagged_description"].to_csv("tagged_description.txt",
                                     sep="\n",
                                     index=False,
                                     header=False)
    
    print("Data preparation complete!")

if __name__ == "__main__":
    prepare_data()
