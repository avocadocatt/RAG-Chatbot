from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
import os
import shutil 

from src.rag_pipeline import RAGPipeline
from src.config import PINECONE_INDEX_NAME

app = FastAPI(
    title="RAG Chatbot API (Gemini + Pinecone)",
    description="API for interacting with a Retrieval Augmented Generation chatbot.",
    version="1.0.0"
)

try:
    rag_pipeline_instance = RAGPipeline()
except Exception as e:
    print(f"CRITICAL: Failed to initialize RAGPipeline: {e}")
    rag_pipeline_instance = None


# --- Pydantic Models for Request/Response Body ---
class IndexRequest(BaseModel):
    documents_path: str = "data/"

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

class IndexStatusResponse(BaseModel):
    index_name: str
    status: str
    vector_count: int
    dimension: int

class MessageResponse(BaseModel):
    message: str

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    global rag_pipeline_instance
    if rag_pipeline_instance is None:
        try:
            print("Attempting to re-initialize RAGPipeline on startup...")
            rag_pipeline_instance = RAGPipeline()
            print("RAGPipeline re-initialized successfully on startup.")
        except Exception as e:
            print(f"CRITICAL: Failed to re-initialize RAGPipeline on startup: {e}")
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory.")
    if not os.listdir("data"):
        example_file_path = os.path.join("data", "example.txt")
        if not os.path.exists(example_file_path):
            with open(example_file_path, "w", encoding="utf-8") as f:
                f.write("Đây là tài liệu ví dụ được tạo tự động cho API.\n")
                f.write("Bạn có thể thêm các file .txt khác vào thư mục 'data' và gọi API /index_documents.")
            print(f"Created '{example_file_path}' as no documents were found in 'data' directory.")


@app.post("/index_documents", response_model=MessageResponse, tags=["Indexing"])
async def index_documents_endpoint(request_body: IndexRequest = Body(IndexRequest())):
    """
    Processes and indexes documents from the specified path.
    If `documents_path` is not provided, it defaults to "data/".
    """
    if rag_pipeline_instance is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized. Check server logs.")

    doc_path = request_body.documents_path
    if not os.path.exists(doc_path) or not os.path.isdir(doc_path):
        raise HTTPException(status_code=400, detail=f"Document path '{doc_path}' does not exist or is not a directory.")
    if not os.listdir(doc_path):
        raise HTTPException(status_code=400, detail=f"Document path '{doc_path}' is empty. Add files to index.")

    try:
        print(f"Starting document indexing from '{doc_path}' via API...")
        rag_pipeline_instance.process_and_index_documents(documents_path=doc_path)
        stats = rag_pipeline_instance.vector_store.index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0) if hasattr(stats, 'get') else getattr(stats, 'total_vector_count', 0) 
        
        return MessageResponse(message=f"Documents from '{doc_path}' processed and indexed successfully. Index '{PINECONE_INDEX_NAME}' now has {vector_count} vectors.")
    except Exception as e:
        print(f"Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Querying"])
async def query_chatbot_endpoint(request_body: QueryRequest):
    """
    Asks a question to the RAG chatbot and gets an answer.
    """
    if rag_pipeline_instance is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized. Check server logs.")
    if not request_body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        print(f"Received query via API: {request_body.question}")
        answer = rag_pipeline_instance.query(request_body.question)
        return QueryResponse(question=request_body.question, answer=answer)
    except Exception as e:
        print(f"Error during query processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.get("/index_status", response_model=IndexStatusResponse, tags=["Indexing"])
async def get_index_status_endpoint():
    """
    Gets the current status and stats of the Pinecone index.
    """
    if rag_pipeline_instance is None or rag_pipeline_instance.vector_store is None or rag_pipeline_instance.vector_store.index is None:
        temp_pipeline_for_status_check = None
        try:
            from src.vector_store import PineconeVectorStore
            from src.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_VECTOR_DIMENSION
            
            import pinecone
            try:
                pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
                if PINECONE_INDEX_NAME in pinecone.list_indexes():
                    index_desc = pinecone.describe_index(PINECONE_INDEX_NAME)
                    index_stats_obj = pinecone.Index(PINECONE_INDEX_NAME).describe_index_stats()
                    
                    vector_count = getattr(index_stats_obj, 'total_vector_count', 0)
                    dimension = getattr(index_stats_obj, 'dimension', PINECONE_VECTOR_DIMENSION)

                    return IndexStatusResponse(
                        index_name=PINECONE_INDEX_NAME,
                        status=index_desc.status.get('state', 'Unknown') if hasattr(index_desc, 'status') and isinstance(index_desc.status, dict) else 'Ready', # Phỏng đoán cấu trúc status
                        vector_count=vector_count,
                        dimension=dimension
                    )
                else:
                    return IndexStatusResponse(
                        index_name=PINECONE_INDEX_NAME,
                        status="Not Found",
                        vector_count=0,
                        dimension=PINECONE_VECTOR_DIMENSION
                    )
            except Exception as init_err:
                 raise HTTPException(status_code=503, detail=f"RAG Pipeline not initialized, and error checking Pinecone status: {init_err}")

        except ImportError:
             raise HTTPException(status_code=503, detail="RAG Pipeline not initialized, and unable to check Pinecone status directly.")
        except Exception as e_check:
            raise HTTPException(status_code=500, detail=f"Error checking Pinecone status directly: {str(e_check)}")


    try:
        index_stats = rag_pipeline_instance.vector_store.index.describe_index_stats()
        
        vector_count = getattr(index_stats, 'total_vector_count', 0)
        dimension = getattr(index_stats, 'dimension', 0)

        status_str = "Ready"
        try:
            import pinecone
            if not hasattr(pinecone, 'api_key') or not pinecone.api_key:
                 pinecone.init(api_key=rag_pipeline_instance.vector_store.api_key, 
                               environment=rag_pipeline_instance.vector_store.environment)
            
            index_description = pinecone.describe_index(PINECONE_INDEX_NAME)
            
            if hasattr(index_description, 'status') and isinstance(index_description.status, dict):
                status_str = index_description.status.get('state', 'Ready')
            elif hasattr(index_description, 'status') and isinstance(index_description.status, str):
                 status_str = index_description.status

        except Exception as desc_err:
            print(f"Could not get detailed index status via describe_index: {desc_err}")


        return IndexStatusResponse(
            index_name=PINECONE_INDEX_NAME,
            status=status_str,
            vector_count=vector_count,
            dimension=dimension
        )
    except Exception as e:
        print(f"Error getting index status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get index status: {str(e)}")


@app.delete("/delete_index", response_model=MessageResponse, tags=["Indexing"])
async def delete_pinecone_index_endpoint(
    confirm: bool = Query(False, description="Set to true to confirm deletion. THIS IS IRREVERSIBLE.")
):
    """
    Deletes the Pinecone index specified in the configuration.
    Requires `confirm=true` query parameter to proceed.
    """
    if rag_pipeline_instance is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized. Check server logs.")
    if not confirm:
        raise HTTPException(status_code=400, detail=f"Deletion not confirmed. Add '?confirm=true' to the URL to delete index '{PINECONE_INDEX_NAME}'. This action is irreversible.")

    try:
        index_name_to_delete = rag_pipeline_instance.vector_store.index_name
        print(f"Attempting to delete Pinecone index '{index_name_to_delete}' via API...")
        rag_pipeline_instance.vector_store.delete_index()
        if rag_pipeline_instance.vector_store:
            rag_pipeline_instance.vector_store.index = None
        return MessageResponse(message=f"Pinecone index '{index_name_to_delete}' has been deleted successfully.")
    except Exception as e:
        print(f"Error deleting index: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete index: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print(f"Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000)