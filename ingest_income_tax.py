import os
from pathlib import Path
from typing import List
from pypdf import PdfReader
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document

PDF_PATH = "data\IncomeTaxOrdinance.pdf"
PERSIST_DIR = "./chroma_db"
USE_CHROMA = True
USE_OPENAI_EMBEDDINGS = False 

def load_pdf_text(pdf_path: str) -> List[Document]:
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        meta = {"source": "IncomeTaxOrdinance.pdf", "page": i + 1}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def chunk_documents(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for d in docs:
        chunks.extend(splitter.split_documents([d]))
    return chunks

def get_embeddings():
    if USE_OPENAI_EMBEDDINGS:
        return OpenAIEmbeddings()
    else:
        # ✅ Uses HuggingFace (PyTorch backend only)
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def persist_vectorstore(chunks: List[Document], persist_dir: str):
    embeddings = get_embeddings()
    if USE_CHROMA:
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
        vectordb.persist()
    else:
        vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

def main():
    path = Path(PDF_PATH)
    if not path.exists():
        raise FileNotFoundError(f"{PDF_PATH} not found.")
    print("📄 Loading PDF...")
    docs = load_pdf_text(str(path))
    print(f"✅ Loaded {len(docs)} pages. Splitting into chunks...")
    chunks = chunk_documents(docs)
    print(f"✅ {len(chunks)} chunks created. Building embeddings...")
    persist_vectorstore(chunks, PERSIST_DIR)
    print("🎉 Done! Vector DB saved in", PERSIST_DIR)

if __name__ == "__main__":
    main()
