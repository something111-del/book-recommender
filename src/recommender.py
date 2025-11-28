from .vector_store import query_books
import pandas as pd

class BookRecommender:
    def __init__(self):
        pass

    def recommend(self, query, category="All", top_k=5):
        """
        Recommends books based on query and category filter.
        """
        print(f"Searching for: '{query}' in category: '{category}'")
        
        results = query_books(
            query_text=query,
            n_results=top_k,
            filter_category=category if category != "All" else None
        )
        
        # Format results
        recommendations = []
        if results and results['ids']:
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            documents = results['documents'][0]
            distances = results['distances'][0]
            
            for i in range(len(ids)):
                rec = {
                    "id": ids[i],
                    "title": metadatas[i].get("title", "Unknown Title"),
                    "authors": metadatas[i].get("authors", "Unknown Author"),
                    "description": documents[i],
                    "category": metadatas[i].get("predicted_category", "Unknown"),
                    "thumbnail": metadatas[i].get("thumbnail", ""),
                    "score": distances[i] # Lower is better for L2, Higher for Cosine depending on impl
                }
                recommendations.append(rec)
                
        return recommendations
