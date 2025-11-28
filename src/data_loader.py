import kagglehub
import pandas as pd
import shutil
import os
from pathlib import Path
from .config import RAW_DATA_DIR, KAGGLE_DATASET

def download_data():
    """
    Downloads the dataset from Kaggle using kagglehub and moves it to the raw data directory.
    """
    print(f"Downloading dataset: {KAGGLE_DATASET}...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset downloaded to cache: {path}")

    # Move files to our raw data directory
    # kagglehub downloads to a cache dir, we want it in our project
    source_dir = Path(path)
    
    for file_path in source_dir.glob("*"):
        if file_path.is_file():
            dest_path = RAW_DATA_DIR / file_path.name
            print(f"Copying {file_path.name} to {dest_path}...")
            shutil.copy2(file_path, dest_path)
            
    print("Data download and setup complete.")

def load_raw_data(filename="books.csv"):
    """
    Loads the raw data into a pandas DataFrame.
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        # Try to find any csv if the specific name isn't found (sometimes names change)
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_files:
            file_path = csv_files[0]
            print(f"Warning: {filename} not found. Loading {file_path.name} instead.")
        else:
            raise FileNotFoundError(f"No CSV file found in {RAW_DATA_DIR}. Did you run download_data()?")
    
    return pd.read_csv(file_path)

if __name__ == "__main__":
    download_data()
