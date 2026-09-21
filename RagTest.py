import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()
embedding=DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

try:
    loader=TextLoader("test.txt",
                      encoding="utf-8")
    document=loader.load()
    print(f"加载了{len(document)}个文档")

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )

    docs=splitter.split_documents(document)
    print(f"切成了{len(docs)}个分片")

    db=Chroma(
        embedding_function=embedding,
        persist_directory='./chroma_db')
    print("存入成功")

    results=db.similarity_search("西南交大录取分数线",k=5)
    for i, doc in enumerate(results):
        print(f"片段{i + 1}: {doc.page_content}")
except Exception as e:
    print(f"出错啦{e}")
