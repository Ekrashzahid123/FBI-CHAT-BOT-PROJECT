import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.staticfiles import StaticFiles

from langchain_core.language_models import FakeListLLM  # lightweight mock LLM

PERSIST_DIR = "./chroma_db"
USE_OPENAI_EMBEDDINGS = False  # we’re using HuggingFace
USE_OPENAI_LLM = False         # disabled, no API key

app = FastAPI(title="Income Tax RAG Chatbot")

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10

class SourceItem(BaseModel):
    page: int
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": req.top_k})

   
    llm = FakeListLLM(responses=["ANSWER IS FOLLOWING."])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    res = chain(req.question)
    answer = res["result"]
    docs = res.get("source_documents", [])

    sources = []
    for d in docs:
        sources.append(SourceItem(
            page=d.metadata.get("page", -1),
            snippet=d.page_content[:400].replace("\n", " ")
        ))

    return QueryResponse(answer=answer, sources=sources)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
