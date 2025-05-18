import google.generativeai as genai
from src.config import GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL
import time

genai.configure(api_key=GOOGLE_API_KEY)

class GeminiEmbeddingClient:
    def __init__(self, model_name=GEMINI_EMBEDDING_MODEL):
        self.model_name = model_name

    def get_embeddings(self, texts, task_type="RETRIEVAL_DOCUMENT", title=None):
        """
        Generates embeddings for a list of texts using Gemini.
        task_type can be: "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY",
                          "CLASSIFICATION", "CLUSTERING".
        """
        if not texts:
            return []
        if isinstance(texts, str):
            texts = [texts]

        embeddings_list = []
        for text_batch in self._batch_texts(texts, batch_size=100): # Gemini API supports batch size up to 100
            try:
                print(f"Generating embeddings for batch of {len(text_batch)} texts...")
                if title:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text_batch,
                        task_type=task_type,
                        title=title
                    )
                else:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text_batch,
                        task_type=task_type
                    )
                embeddings_list.extend(result['embedding'])
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                embeddings_list.extend([None] * len(text_batch))
                time.sleep(1) # Time for retrying after error
        return embeddings_list

    def get_embedding(self, text, task_type="RETRIEVAL_QUERY", title=None):
        """Generates embedding for a single text."""
        if not text:
            return None
        try:
            if title:
                 result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=task_type,
                    title=title
                )
            else:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=task_type
                )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding for '{text[:50]}...': {e}")
            return None

    def _batch_texts(self, texts, batch_size=100):
        """Yield successive batch_size-sized chunks from texts."""
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]
