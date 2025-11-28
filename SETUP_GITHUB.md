# Setup Guide for GitHub Upload

This guide will help you prepare and upload your book recommender project to GitHub.

## Pre-Upload Checklist

### ✅ Files Organized

The project has been organized into the following structure:

```
book-recommender/
├── .gitignore                          # Git ignore rules
├── .env.example                        # Example environment file
├── LICENSE                             # MIT License
├── README.md                           # Main documentation
├── CONTRIBUTING.md                     # Contribution guidelines
├── requirements.txt                    # Python dependencies
├── data/
│   ├── raw/.gitkeep                   # Raw data directory (empty for GitHub)
│   ├── processed/.gitkeep             # Processed data directory
│   └── chroma_db/                     # Vector database (gitignored)
├── notebooks/
│   ├── data-exploration.ipynb         # Data cleaning notebook
│   ├── text-classification.ipynb      # Zero-shot classification
│   ├── sentiment-analysis.ipynb       # Emotion analysis
│   ├── vector-search.ipynb            # Semantic search
│   └── executed/.gitkeep              # Executed notebooks (gitignored)
├── scripts/
│   ├── zero_shot_classification.py    # Standalone demo
│   └── gradio-dashboard-simple.py     # Simplified dashboard
├── outputs/.gitkeep                    # Output files (gitignored)
├── gradio-dashboard.py                 # Full-featured dashboard
├── src/                                # Source code modules
└── tests/                              # Test files
```

### ✅ Large Files Excluded

The following large files are excluded via `.gitignore`:
- `*.csv` files (datasets)
- `*.nbconvert.ipynb` (executed notebooks)
- Model cache files (`.pt`, `.pth`, `.bin`)
- Virtual environment (`.venv/`)
- Environment variables (`.env`)
- Database files (`data/chroma_db/`)

### ✅ Environment Variables

Create a `.env.example` file for users:

```bash
# OpenAI API Key (required for semantic search)
OPENAI_API_KEY=your_api_key_here

# Optional: Model cache directory
HF_HOME=.cache/huggingface
```

## Upload Steps

### 1. Initialize Git Repository (if not already done)

```bash
cd /Users/karepallimahesh/Desktop/py/LLM/book-recommender
git init
```

### 2. Add Files to Git

```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

### 3. Create Initial Commit

```bash
git commit -m "Initial commit: Book Recommender with BART-MNLI zero-shot classification"
```

### 4. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `book-recommender` (or your preferred name)
3. Description: "LLM-powered book recommendation system using zero-shot classification and semantic search"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 5. Connect to GitHub

```bash
# Add remote repository (replace with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/book-recommender.git

# Verify remote
git remote -v
```

### 6. Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Post-Upload Tasks

### 1. Add Repository Topics

On GitHub, add relevant topics:
- `machine-learning`
- `nlp`
- `transformers`
- `zero-shot-learning`
- `book-recommendation`
- `gradio`
- `bart-mnli`
- `semantic-search`

### 2. Create Release (Optional)

1. Go to "Releases" → "Create a new release"
2. Tag version: `v1.0.0`
3. Release title: "Initial Release - Zero-Shot Book Classifier"
4. Description: Highlight key features

### 3. Enable GitHub Pages (Optional)

If you want to host documentation:
1. Settings → Pages
2. Source: Deploy from branch
3. Branch: main, folder: /docs

### 4. Add Badges to README

The README already includes badges for:
- Python version
- Transformers
- Gradio

### 5. Set Up GitHub Actions (Optional)

Create `.github/workflows/tests.yml` for automated testing:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
```

## Important Notes

### What's Included in GitHub

✅ Source code (notebooks, scripts)
✅ Documentation (README, CONTRIBUTING)
✅ Configuration files (requirements.txt, .gitignore)
✅ License
✅ Empty directory structure

### What's NOT Included (Gitignored)

❌ Large CSV files (users download separately)
❌ Model cache files
❌ Virtual environment
❌ API keys and secrets
❌ Executed notebook outputs
❌ Vector database files

### For Users Cloning Your Repository

Users will need to:

1. Clone the repository
2. Create virtual environment
3. Install dependencies
4. Download the dataset (provide link in README)
5. Set up `.env` file with API keys
6. Run notebooks or scripts

## Updating Your Repository

After making changes:

```bash
# Check status
git status

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add: feature description"

# Push to GitHub
git push origin main
```

## Troubleshooting

### Large File Error

If you accidentally try to commit large files:

```bash
# Remove from staging
git reset HEAD large_file.csv

# Add to .gitignore
echo "large_file.csv" >> .gitignore
```

### Sensitive Data Committed

If you accidentally commit API keys:

```bash
# Remove from history (use with caution)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push
git push origin --force --all
```

Then immediately rotate your API keys!

## Next Steps

1. ✅ Review all files before pushing
2. ✅ Test clone on another machine
3. ✅ Update README with your GitHub username
4. ✅ Add screenshots to README (optional)
5. ✅ Share your repository!

---

**Ready to upload?** Follow the steps above and your project will be live on GitHub! 🚀
