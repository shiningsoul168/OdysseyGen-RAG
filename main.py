
from fastapi import FastAPI
from pydantic import BaseModel,Field

from rag_core import search

app=FastAPI(title="OdysseyGen RAG Service")

class RagRequest(BaseModel):
    query: str=Field(...,description="Query string")
    top_k: int=Field(2,ge=1,le=10,description="Top K words")

class RagResponse(BaseModel):
    code: int=Field(...,description="Code")
    context: list[str]
    count: int

@app.post("/rag/retrieve",response_model=RagResponse)
async def rag_retrieve(request: RagRequest):
    try:
        result=search(request.query,request.top_k)
        return RagResponse(code=200,context=result,count=len(result))
    except Exception as e:
        print(f"RAG 检索失败: {e}")
        return RagResponse(code=500,context=[],count=0)