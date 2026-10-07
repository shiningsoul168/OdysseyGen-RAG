# -*- coding: utf-8 -*-
"""
rag_eval.py 的本机配置 —— 单独成文件，脚本本体升级时不会覆盖它。

★ 以后要改配置，只改这里，不要改 rag_eval.py 顶部的默认值。

已核实（2026-09-27，只读探测 chroma.sqlite3）：
    DB_DIR = ./chroma_db      一个库，两个 collection
    kaoyan  69 条向量
    kaogong 5649 条向量
    （另有一个空的 langchain collection，是脚本误用默认库时自动建的，可删）
"""

# data_type -> collection 名（不是目录名！两个 collection 在同一个 DB_DIR 里）
COLLECTIONS = {
    "kaoyan": "kaoyan",
    "kaogong": "kaogong",
}

# 分类型拒答阈值：None = 只报告不定阈值（先看标定表的分布再填）
# ⚠ L2 与 cosine 的距离不可跨空间比较，换空间后必须重新标定
THRESHOLD = {
    "kaoyan": None,
    "kaogong": None,
}
