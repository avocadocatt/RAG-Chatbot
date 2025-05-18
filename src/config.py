import os
from dotenv import load_dotenv

load_dotenv()

# -- LLM Configuration --
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL")

# -- Pinecone Configuration --
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_VECTOR_DIMENSION = 768

# -- Document Processing Configuration --
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# -- RAG Configuration --
TOP_K_RESULTS = 5