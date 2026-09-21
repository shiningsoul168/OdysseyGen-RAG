import os;

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

DB_DIR=os.getenv("DB_DIR")
embeddings=DashScopeEmbeddings(model="text-embedding-v1")
vectordb=Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

def search(query:str,top_k:int=2) ->list[str]:
    """检索返回top_k片段"""
    results=vectordb.similarity_search(query,k=top_k)
    return [doc.page_content for doc in results]