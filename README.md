# RAG-Chatbot
This project implements a Retrieval Augmented Generation (RAG) chatbot powered by Google's Gemini models for embeddings and generation, and Pinecone as the vector database. It allows users to query information from their own documents by first indexing them into Pinecone and then using an LLM to generate answers based on retrieved context. The application is exposed via a FastAPI interface.

Install `python 3.12` from [Link download Python 3.12 version](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe).

## Install needed library
Run this command on `terminal` to install requirements library.
```python
pip install -r requirements.txt
```

## Run
Run this command on `terminal` to start API with localhost:
```python
python app.py
```

## API url
1. Indexing document:
```python
http://localhost:8000/index_documents
payload = {
    documents_path = "data/" # optional, default: "data/"
}
response = {
    message: "Indexing completed"
}
```
2. Get index status:
```python
http://localhost:8000/index_status
response = {
    index_name = "vietdb",
    status = "READY",
    vector_count = 167,
    dimension = 768
}
```
3. Delete index:
```python
http://localhost:8000/delete_index
response = {
    message = "Deleting completed"
}
```
4. Query:
```python
http://localhost:8000/query
payload = {
    question = "What is RAG chatbot?"
}
response = {
    question = "What is RAG chatbot?",
    answer = "A RAG chatbot refers to a chatbot that uses the Retrieval-Augmented Generation (RAG) technique to improve its ability to generate accurate, contextually relevant, and up-to-date responses"
}
```