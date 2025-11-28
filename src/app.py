import gradio as gr
from .recommender import BookRecommender
from .config import PROCESSED_DATA_DIR
import pandas as pd

# Initialize Recommender
recommender = BookRecommender()

def get_categories():
    # Load categories from processed data if available, else default
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "books_processed.csv")
        if 'predicted_category' in df.columns:
            cats = sorted(df['predicted_category'].unique().tolist())
            return ["All"] + cats
    except Exception:
        pass
    return ["All", "Fiction", "Science Fiction", "Fantasy", "Mystery", "Thriller", "Romance", "History", "Biography", "Business", "Science", "Technology"]

def search_books(query, category):
    if not query:
        return []
    
    recommendations = recommender.recommend(query, category)
    
    # Format for Gradio Gallery or HTML
    # Let's return HTML for better control over layout
    html_output = ""
    for book in recommendations:
        thumbnail = book['thumbnail']
        if not thumbnail or thumbnail == 'nan':
            thumbnail = "https://via.placeholder.com/128x192?text=No+Cover"
            
        html_output += f"""
        <div style="display: flex; margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 8px;">
            <img src="{thumbnail}" style="width: 100px; height: 150px; object-fit: cover; margin-right: 20px;">
            <div>
                <h3 style="margin-top: 0;">{book['title']}</h3>
                <p><strong>Author:</strong> {book['authors']}</p>
                <p><strong>Category:</strong> {book['category']}</p>
                <p style="font-size: 0.9em; color: #555;">{book['description'][:300]}...</p>
            </div>
        </div>
        """
    if not html_output:
        html_output = "<p>No recommendations found.</p>"
        
    return html_output

# UI Layout
with gr.Blocks(title="LLM Semantic Book Recommender") as demo:
    gr.Markdown("# 📚 LLM Semantic Book Recommender")
    gr.Markdown("Find your next read using natural language search.")
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="What are you looking for?", placeholder="e.g., A book about space exploration and aliens")
            category_dropdown = gr.Dropdown(choices=get_categories(), value="All", label="Filter by Category")
            search_btn = gr.Button("Recommend Books", variant="primary")
        
        with gr.Column(scale=2):
            results_output = gr.HTML(label="Recommendations")
            
    search_btn.click(fn=search_books, inputs=[query_input, category_dropdown], outputs=results_output)
    query_input.submit(fn=search_books, inputs=[query_input, category_dropdown], outputs=results_output)

if __name__ == "__main__":
    demo.launch()
