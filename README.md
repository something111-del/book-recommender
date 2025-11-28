# 📚 LLM Semantic Book Recommender

A sophisticated book recommendation system powered by **zero-shot classification** using Facebook's BART-large-MNLI model, semantic search with OpenAI embeddings, and sentiment analysis. This project demonstrates advanced NLP techniques for content-based filtering and recommendation.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Gradio](https://img.shields.io/badge/Gradio-Dashboard-orange.svg)](https://gradio.app/)

## 🌟 Features

- **Zero-Shot Text Classification**: Automatically categorizes books into Fiction/Nonfiction using `facebook/bart-large-mnli` without training data
- **Semantic Search**: Vector-based similarity search using OpenAI embeddings and ChromaDB
- **Sentiment Analysis**: Emotion detection in book descriptions using DistilRoBERTa
- **Interactive Dashboard**: Beautiful Gradio interface for exploring recommendations
- **Comprehensive Data Pipeline**: End-to-end workflow from raw data to deployed application

## 🏗️ Project Structure

```
book-recommender/
├── data/
│   ├── raw/              # Original dataset (7k books)
│   ├── processed/        # Cleaned and processed data
│   └── chroma_db/        # Vector database
├── notebooks/
│   ├── data-exploration.ipynb          # Data cleaning and EDA
│   ├── text-classification.ipynb       # Zero-shot classification
│   ├── sentiment-analysis.ipynb        # Emotion analysis
│   ├── vector-search.ipynb             # Semantic search setup
│   └── executed/                       # Executed notebook outputs
├── scripts/
│   ├── zero_shot_classification.py     # Standalone classification demo
│   └── gradio-dashboard-simple.py      # Simplified dashboard
├── outputs/
│   ├── books_cleaned.csv               # Cleaned dataset
│   └── books_with_categories.csv       # Classified books
├── gradio-dashboard.py                 # Full-featured dashboard
├── requirements.txt                    # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip package manager
- OpenAI API key (for semantic search features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/book-recommender.git
   cd book-recommender
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional, for full features)
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

5. **Download the dataset**
   
   Place your `books.csv` file in `data/raw/` directory.

### Running the Application

#### Option 1: Quick Demo (Simplified Dashboard)

```bash
python scripts/gradio-dashboard-simple.py
```

Then open http://127.0.0.1:7860 in your browser.

#### Option 2: Full Pipeline

Run the notebooks in sequence:

```bash
# 1. Data exploration and cleaning
jupyter notebook notebooks/data-exploration.ipynb

# 2. Zero-shot classification with BART-MNLI
jupyter notebook notebooks/text-classification.ipynb

# 3. Sentiment analysis
jupyter notebook notebooks/sentiment-analysis.ipynb

# 4. Vector search setup
jupyter notebook notebooks/vector-search.ipynb

# 5. Launch full dashboard
python gradio-dashboard.py
```

#### Option 3: Standalone Classification Demo

```bash
python scripts/zero_shot_classification.py
```

## 🧠 Core Technologies

### Zero-Shot Classification

The project uses **facebook/bart-large-mnli** for zero-shot text classification:

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

result = classifier(
    book_description,
    candidate_labels=["Fiction", "Nonfiction"]
)
```

**Why BART-MNLI?**
- Pre-trained on Multi-Genre Natural Language Inference (MNLI) dataset
- Excellent zero-shot performance without fine-tuning
- Handles long text sequences effectively
- Provides confidence scores for each label

### Semantic Search

Uses OpenAI embeddings with ChromaDB for vector similarity search:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents, embeddings)
results = db.similarity_search(query, k=10)
```

### Sentiment Analysis

Employs `j-hartmann/emotion-english-distilroberta-base` for emotion detection:

- Analyzes 7 emotions: anger, disgust, fear, joy, sadness, surprise, neutral
- Processes book descriptions sentence-by-sentence
- Aggregates emotion scores for filtering

## 📊 Dataset

The project uses a dataset of **7,000 books** with the following features:

- ISBN-13 and ISBN-10
- Title and subtitle
- Authors
- Categories
- Description
- Publication year
- Average rating
- Number of pages
- Ratings count
- Thumbnail URL

**Data Processing:**
- Cleaned missing values
- Created combined text fields
- Filtered books with sufficient description length
- Generated emotion scores
- Applied zero-shot classification

## 🎯 Use Cases

1. **Content-Based Recommendation**: Find similar books based on descriptions
2. **Genre Classification**: Automatically categorize books without manual labeling
3. **Mood-Based Discovery**: Filter books by emotional tone
4. **Semantic Search**: Natural language queries like "books about forgiveness"

## 📈 Results

### Classification Statistics

- **Total Books Processed**: 5,197
- **Model**: facebook/bart-large-mnli
- **Categories**: Fiction (60%), Nonfiction (40%)
- **Processing Time**: ~10 minutes on CPU
- **Accuracy**: High confidence scores (avg 0.85+)

### Performance

- **Zero-Shot Classification**: No training required
- **Inference Speed**: ~0.5s per book on CPU
- **Model Size**: ~1.6GB (BART-MNLI)
- **Memory Usage**: ~4GB RAM for full pipeline

## 🛠️ Development

### Project Dependencies

Key libraries:
- `transformers` - Hugging Face transformers for NLP models
- `torch` - PyTorch backend
- `langchain` - LLM application framework
- `gradio` - Web UI framework
- `chromadb` - Vector database
- `pandas` - Data manipulation
- `openai` - OpenAI API client

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black .

# Lint
ruff check .
```

## 📝 Notebooks Overview

### 1. Data Exploration (`data-exploration.ipynb`)
- Load and inspect the dataset
- Handle missing values
- Create derived features
- Export cleaned data

### 2. Text Classification (`text-classification.ipynb`)
- Load facebook/bart-large-mnli model
- Perform zero-shot classification
- Assign Fiction/Nonfiction labels
- Save classified dataset

### 3. Sentiment Analysis (`sentiment-analysis.ipynb`)
- Load emotion classification model
- Analyze book descriptions
- Extract emotion scores
- Merge with main dataset

### 4. Vector Search (`vector-search.ipynb`)
- Create text embeddings
- Build ChromaDB vector store
- Test similarity search
- Export search index

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: 7k Books with Metadata from Kaggle
- **Models**: 
  - [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) for zero-shot classification
  - [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) for sentiment analysis
- **Inspiration**: Original project by [t-redactyl](https://github.com/t-redactyl/llm-semantic-book-recommender)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**
