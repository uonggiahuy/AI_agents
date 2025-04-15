from langchain.text_splitter import RecursiveCharacterTextSplitter #CharacterTextSplitter:dùng với 1 đoạn văn bản
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from pathlib import Path

pdf_data_path = Path("E:/Code programs/n2025/AI agents/AI_agents/data")
vector_db_path = Path("E:/Code programs/n2025/AI agents/AI_agents/vector_store/db_faiss")

def create_db_from_pdf():
    # goi loader de quet het thong tin trong data
    loader = DirectoryLoader(pdf_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedidng_model = GPT4AllEmbeddings(model_file=Path("E:/Code programs/n2025/AI agents/AI_agents/model/all-MiniLM-L6-v2-f16.gguf"))
    db = FAISS.from_documents(chunks, embedidng_model)
    db.save_local(vector_db_path)
    
    return db

create_db_from_pdf()
