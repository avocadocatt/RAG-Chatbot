# RAG-Chatbot
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
    documents_path = "data/"
}
```
2. Get index status:
```python
http://localhost:8000/index_status
```
3. Delete index:
```python
http://localhost:8000/delete_index
```
4. Query:
```python
http://localhost:8000/query
payload = {
    question = "What is RAG chatbot?"
}
```