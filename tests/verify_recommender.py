from src.recommender import BookRecommender
import sys

def test_recommender():
    print("Initializing Recommender...")
    try:
        rec = BookRecommender()
    except Exception as e:
        print(f"Failed to initialize recommender: {e}")
        sys.exit(1)
        
    query = "space exploration"
    print(f"Testing recommendation for query: '{query}'")
    results = rec.recommend(query, top_k=3)
    
    if not results:
        print("No results found.")
    else:
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"- {r['title']} ({r['category']})")
            
    print("Verification complete.")

if __name__ == "__main__":
    test_recommender()
