"""RAG 检索核心 —— P2 接线版"""
import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from rag_intent import parse_intent, build_where_document

load_dotenv()

logger = logging.getLogger(__name__)

DB_DIR = os.getenv("DB_DIR") or "./chroma_db"
embeddings = DashScopeEmbeddings(model="text-embedding-v4")

# 白名单：绝不把入参当 collection 名（LangChain Chroma 是 get_or_create，会静默建空库）
COLLECTION_WHITELIST = {"kaoyan": "kaoyan", "kaogong": "kaogong"}

# Chroma handle 缓存：每次 new 都读 sqlite + 载 HNSW，2s 超时扛不住
_HANDLES = {}


def _handle(data_type):
    """返回 Chroma handle 或 None（未知 data_type / 库不存在）"""
    name = COLLECTION_WHITELIST.get(data_type)
    if name is None:
        return None
    if name in _HANDLES:
        return _HANDLES[name]
    try:
        h = Chroma(persist_directory=DB_DIR,
                   embedding_function=embeddings,
                   collection_name=name)
        if h._collection.count() == 0:
            return None
        _HANDLES[name] = h
        return h
    except Exception as e:
        logger.warning("打开 collection %s 失败: %s", name, e)
        return None


def _empty(strategy="reject", reason="", warnings=None):
    return {
        "strategy": strategy,
        "reason": reason,
        "count": 0,
        "context": [],
        "metadatas": [],
        "warnings": warnings or [],
    }


def search(query: str, top_k: int = 2, data_type: str = "kaoyan") -> dict:
    """P2 检索入口。返回 dict（strategy/reason/count/context/metadatas/warnings）"""
    db = _handle(data_type)
    if db is None:
        return _empty("reject", f"未支持或不存在的数据类型：{data_type}")

    # ---- 意图解析（异常降级为 semantic，但记录 + 打标）----
    try:
        intent = parse_intent(query, data_type)
        degraded = False
    except Exception as e:
        logger.exception("parse_intent 异常，降级 semantic")
        intent = {
            "where": None, "contains": None, "contains_op": "or",
            "strategy": "semantic", "unsupported": [], "warning": [],
        }
        degraded = True

    strategy = intent["strategy"]
    where = intent.get("where")
    contains = intent.get("contains")
    contains_op = intent.get("contains_op", "or")
    where_document = build_where_document(contains, contains_op)

    # ---- 解析期 reject ----
    if strategy == "reject":
        reason = "；".join(intent.get("unsupported") or []) or "该问题库内无匹配条件"
        return _empty("reject", reason)

    # ---- 执行期 count（零 embedding）----
    total = None
    if where or where_document:
        ck = {"include": []}
        if where:
            ck["where"] = where
        if where_document:
            ck["where_document"] = where_document
        try:
            total = len(db._collection.get(**ck)["ids"])
        except Exception as e:
            logger.warning("count 查询失败: %s", e)
            total = None

    # ---- 执行期 finalize：filtered 但 0 条 → reject ----
    if strategy == "filtered" and total == 0:
        return _empty("reject", "过滤后无匹配数据")

    # ---- 向量检索 ----
    emb = db._embedding_function.embed_query(query)
    kwargs = {
        "query_embeddings": [emb],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    if where_document:
        kwargs["where_document"] = where_document

    try:
        res = db._collection.query(**kwargs)
    except Exception as e:
        logger.exception("向量检索失败")
        return _empty("reject", f"检索失败：{e}")

    ids = (res.get("ids") or [[]])[0]
    context = res["documents"][0] if ids else []
    metas = res["metadatas"][0] if ids else []

    # ---- count 语义按 strategy 三分支 ----
    if strategy == "semantic":
        count = len(context)              # 语义检索没有"总数"
    elif strategy == "filtered":
        count = total if total is not None else len(context)
    else:
        count = 0

    # ---- warnings ----
    warnings = list(intent.get("warning") or [])
    if degraded:
        warnings.append("解析异常，已降级为语义检索")
    if strategy == "semantic":
        warnings.append("未识别到结构化条件，以下为语义相近结果，未按条件筛选")
    if strategy == "filtered" and total:
        try:
            db_total = db._collection.count()
            if db_total and total / db_total > 0.5:
                warnings.append(f"条件过宽，命中 {total}/{db_total}（{total/db_total:.0%}）")
        except Exception:
            pass

    return {
        "strategy": strategy,
        "reason": "",
        "count": count,
        "context": context,
        "metadatas": metas,
        "warnings": warnings,
    }