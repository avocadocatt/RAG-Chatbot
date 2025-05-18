from src.document_processor import load_documents_from_directory, split_text_into_chunks
from src.embedding_client import GeminiEmbeddingClient
from src.vector_store import PineconeVectorStore
from src.llm import GeminiLLMHandler
from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_VECTOR_DIMENSION
)
import os
from tqdm import tqdm

class RAGPipeline:
    def __init__(self):
        self.embedding_client = GeminiEmbeddingClient()
        self.vector_store = PineconeVectorStore(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            dimension=PINECONE_VECTOR_DIMENSION
        )
        self.llm_handler = GeminiLLMHandler()

    def process_and_index_documents(self, documents_path="data/"):
        """
        Loads documents, splits them, generates embeddings, and upserts to Pinecone.
        """
        print(f"Loading documents from: {documents_path}")
        raw_documents = load_documents_from_directory(documents_path)
        if not raw_documents:
            print("No documents found or loaded.")
            return

        print(f"Splitting {len(raw_documents)} documents into chunks...")
        chunks = split_text_into_chunks(raw_documents, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print("No chunks created from documents.")
            return

        print(f"Total chunks created: {len(chunks)}")
        print("Generating embeddings for chunks...")

        # Embedding for each chunk
        texts_to_embed = [chunk['text'] for chunk in chunks]
        embeddings = []
        batch_size_embedding = 100 # Gemini API supports up to 100 texts per request
        for i in tqdm(range(0, len(texts_to_embed), batch_size_embedding), desc="Embedding chunks"):
            batch_texts = texts_to_embed[i:i + batch_size_embedding]
            batch_embeddings = self.embedding_client.get_embeddings(batch_texts, task_type="RETRIEVAL_DOCUMENT")
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
            else:
                # Handle case where embedding could not be created for batch
                # (e.g., fill with None and filter out later, or stop)
                print(f"Warning: Could not generate embeddings for a batch starting at index {i}")
                embeddings.extend([None] * len(batch_texts))
        # Ensure all chunks have embeddings
        vectors_to_upsert = []
        for i, chunk_data in enumerate(chunks):
            if embeddings[i] is not None:
                vectors_to_upsert.append({
                    "id": chunk_data['id'],
                    "values": embeddings[i],
                    "metadata": {**chunk_data['metadata'], "text_chunk": chunk_data['text']}
                })
            else:
                print(f"Skipping chunk {chunk_data['id']} due to missing embedding.")

        if not vectors_to_upsert:
            print("No valid vectors with embeddings to upsert.")
            return

        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        self.vector_store.upsert_vectors(vectors_to_upsert, batch_size=100) # Pinecone có thể xử lý batch lớn hơn
        print("Document processing and indexing complete.")
        print(f"Pinecone index stats: {self.vector_store.index.describe_index_stats()}")


    def query(self, user_question):
        """
        Takes a user question, retrieves relevant context, and generates an answer.
        """
        print(f"\nUser question: {user_question}")

        # 1. Embed the user question
        print("Embedding user question...")
        query_embedding = self.embedding_client.get_embedding(user_question, task_type="RETRIEVAL_QUERY")
        if not query_embedding:
            return "Xin lỗi, tôi không thể xử lý câu hỏi của bạn vào lúc này (lỗi embedding)."

        # 2. Retrieve relevant chunks from Pinecone
        print(f"Retrieving top-{TOP_K_RESULTS} relevant documents from Pinecone...")
        retrieved_matches = self.vector_store.query_vectors(query_embedding, top_k=TOP_K_RESULTS)

        if not retrieved_matches:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi của bạn."

        # 3. Format context for LLM
        context_parts = []
        print("\n---Retrieved Context Chunks---")
        for i, match in enumerate(retrieved_matches):
            if 'metadata' in match and 'text_chunk' in match.metadata:
                text = match.metadata['text_chunk']
                source = match.metadata.get('source', 'N/A')
                score = match.score
                print(f"Chunk {i+1} (Source: {source}, Score: {score:.4f}):\n{text[:200]}...") # In ra 200 ký tự đầu
                context_parts.append(text)
            else:
                 print(f"Chunk {i+1} (ID: {match.id}) metadata missing text_chunk.")


        if not context_parts:
            return "Xin lỗi, tôi đã tìm thấy các mục liên quan nhưng không thể trích xuất nội dung để trả lời."

        context_for_llm = "\n\n---\n\n".join(context_parts)

        # 4. Generate answer using LLM
        print("\nGenerating answer with LLM...")
        answer = self.llm_handler.generate_answer(user_question, context_for_llm)
        return answer