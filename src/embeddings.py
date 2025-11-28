from openai import OpenAI
from .config import OPENAI_API_KEY, EMBEDDING_MODEL
import time

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model=EMBEDDING_MODEL):
    """
    Generates embedding for a given text using OpenAI API.
    """
    text = text.replace("\n", " ")
    try:
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Simple retry logic or return None
        time.sleep(1)
        try:
            return client.embeddings.create(input=[text], model=model).data[0].embedding
        except Exception as e:
            print(f"Retry failed: {e}")
            return None

def get_embeddings_batch(texts, model=EMBEDDING_MODEL, batch_size=100):
    """
    Generates embeddings for a list of texts in batches.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Clean newlines
        batch = [t.replace("\n", " ") for t in batch]
        try:
            response = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Fallback to single processing or skip?
            # For now, just try to continue or re-raise
            raise e
    return embeddings
