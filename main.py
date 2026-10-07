import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_core import search

logger = logging.getLogger("uvicorn.error")   # 走 uvicorn 的 logger，不然 INFO 不显示


@asynccontextmanager
async def lifespan(app):
    # 预热：① 加载 HNSW ② 建立句柄 ③ 走一次 filtered 分支（覆盖 count）
    # 每次真实请求仍要调远程 embedding（100~500ms），预热消除不掉这部分
    try:
        t0 = time.time()
        search("成都的岗位", top_k=1, data_type="kaogong")   # filtered + count
        logger.info("RAG 预热[kaogong/filtered]完成，耗时 %.2fs", time.time() - t0)

        t0 = time.time()
        search("分数线", top_k=1, data_type="kaoyan")         # 覆盖 kaoyan 句柄
        logger.info("RAG 预热[kaoyan]完成，耗时 %.2fs", time.time() - t0)

        logger.info("RAG 预热全部完成")
    except Exception as e:
        logger.warning("RAG 预热失败（不阻断启动）: %s", e)
    yield


app = FastAPI(title="OdysseyGen RAG Service", lifespan=lifespan)


class RagRequest(BaseModel):
    query: str = Field(..., description="Query String")
    top_k: int = Field(2, ge=1, le=10, description="Top K")
    data_type: str | None = Field(None, description="kaoyan / kaogong")


class RagResponse(BaseModel):
    code: int = Field(..., description="Code")
    strategy: str = ""
    count: int = 0
    context: list[str] = []
    metadatas: list[dict] = []
    warnings: list[str] = []
    reason: str = ""


@app.post("/rag/retrieve", response_model=RagResponse)
async def rag_retrieve(request: RagRequest):
    if not request.data_type:
        return RagResponse(code=200, strategy="reject", reason="未指定 data_type")
    try:
        r = search(request.query, top_k=request.top_k, data_type=request.data_type)
        return RagResponse(code=200, **r)
    except Exception as e:
        logger.exception("RAG 检索失败")
        return RagResponse(code=500, reason=str(e))