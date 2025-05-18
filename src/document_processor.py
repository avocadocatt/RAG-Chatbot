from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import os


def load_documents_from_directory(directory_path):
    """
    Load all .txt files from a directory.
    """
    documents_content = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents_content.append({"name": filename, "content": f.read()})
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        # elif filename.endswith(".pdf"):
        #     loader = PyPDFLoader(os.path.join(directory_path, filename))
        #     pages = loader.load_and_split() # Đây đã là list các Document object của Langchain
        #     # Cần điều chỉnh để phù hợp với cấu trúc `documents_content`
        #     for i, page in enumerate(pages):
        #          documents_content.append({"name": f"{filename}_page_{i+1}", "content": page.page_content})
    return documents_content

def split_text_into_chunks(text_content, chunk_size, chunk_overlap):
    """
    Splits a list of text documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    for doc in text_content:
        doc_chunks = text_splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "id": f"{doc['name']}_chunk_{i}",
                "text": chunk_text,
                "metadata": {"source": doc["name"]}
            })
    return chunks