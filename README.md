# OdysseyGen-RAG

> **为 [OdysseyGen](https://github.com/shiningsoul168/OdysseyGen) 提供真实数据支撑的检索微服务。**
>
> OdysseyGen 是一个职业规划平台。它的核心主张是"让人看清每条路真实的门槛与代价"，
> 所以规划里的每条结论都必须**有据可依、可溯源**，而不是模型编的模板。
> 这个服务负责把真实的招生数据与岗位数据变成可检索的依据。

---

## 技术栈

`FastAPI` · `ChromaDB` · `LangChain` · `DashScope Embedding (text-embedding-v4)` · 双 collection + cosine 空间 · Java 主服务 HTTP 集成（2s 超时 + 失败降级）

## 数据

| collection | 条数 | 内容 |
|---|---|---|
| `kaoyan` | 69 | 四川大学 2025/2026 考研：复试分数线 / 考试科目 / 招生目录 / 报录数据 |
| `kaogong` | 5649 | 四川省 2026 上半年事业单位公开招聘岗位（市州 + 省属 + 中小学教师） |

按业务行类型分四类入库（`分数线+招生目录` / `分数线` / `招生目录` / `报录`），content 带字段标签与【本条含】声明，metadata 承载可过滤字段（学院 / 学位类型 / 年份 / 城市 / 学历门槛 / 政治面貌 / 基层要求…）。

## 核心设计

**1. 意图解析优先于向量检索**

```
query → parse_intent → {where, contains, contains_op, strategy, unsupported, warning}
      ├─ 字段类（党员/基层/学历/城市）→ metadata `where` 过滤（多条件 $and）
      ├─ 实体类（单位名/代码）        → metadata 精确匹配 → 别名归一化 → contains 兜底
      └─ 复合类                        → where + where_document 同时下发
→ 过滤结果为空 ⇒ 返回"暂无数据"（拒答不依赖距离阈值）
→ 非空 ⇒ 过滤集内向量排序取 Top-K，返回 content + metadata 供标注来源
```

**2. 拒答机制：结构化过滤为空（零阈值）**

实测发现**全局相似度阈值无解**：真命中的「党员岗位」Top1 距离 0.84，而最该拒答的「单位没有这个岗」只有 0.49 —— **正确答案比错误答案更远**；换到 cosine 空间重测依然重叠（0.46 vs 0.28）。
→ 结论：**距离只能用于排序，不能用于拒绝**。拒答改走"结构化过滤结果为空"。

**3. 别名归一化**

`contains "四川省水利厅"` 命中 **1** 条，`contains "省水利厅"` 命中 **123** 条，而 gold 就是 123。
→ 不做归一化**召回损失 99%**。归一化采用「前缀剥离 + 值域校验」而非通用字符串剥离（后者会把 `四川大学` 切成 `大学`）。

## 评测（三层，全部零 embedding、秒级）

| 命令 | 覆盖 | 结果 |
|---|---|---|
| `--check` | 40 条检索用例的 metadata 过滤条数 vs gold | **40/40** |
| `--parse-check` | **30 条**意图解析回归（解析层 + 执行层双层断言） | **30/30** |
| `--mode filtered` | 40 条检索用例端到端（过滤 + 过滤集内排序） | **31/31 命中**，9/9 正确拒答 |

**关键对比**（`eval_result_*.md`）：

| 阶段 | 字段类 Hit@1 | 整体纯度 |
|---|---|---|
| `v0-baseline`（单库 / L2 / 纯向量） | 0.33 | 0.39 |
| `v1-cosine`（双 collection / cosine / 纯向量） | 0.33 | 0.40 |
| **`v2-filtered`（+ 意图解析与结构化过滤）** | **1.00** | **1.000** |

> 评测集按问题类型分开取指标：排序类看 Hit@1/MRR，过滤类看纯度与过滤条数一致性，拒答类看正确拒答率。
> 早期用 P@3 时所有 query 都是 0.33 —— 因为库里多数 query 只有 1~3 条相关文档，**P@3 的上限被结构性锁死，分辨不出好坏**。

## 运行

```bash
uv sync                       # 依赖
cp .env.example .env          # 填入 DASHSCOPE_API_KEY / DB_DIR
uv run python export_domains.py   # 导出值域 value_domains.json
uv run uvicorn main:app --host 127.0.0.1 --port 8000

# 评测（都不调用 embedding）
uv run python rag_eval.py --check
uv run python rag_eval.py --parse-check
uv run python rag_eval.py --smoke          # Chroma 算子支持性验证
uv run python rag_eval.py --audit          # metadata 取值分布审计
```

`POST /rag/retrieve`：入参 `{query, top_k, data_type}`，返回 `{code, strategy, count, context, metadatas, warnings, reason}`。
`count` 是过滤命中总数（用于"共 734 个岗位"），`context` 是过滤集内 Top-K（注入 Prompt）。

## 数据来源与合规

- 考研数据：四川大学研究生院公开的招生目录 / 分数线信息
- 岗位数据：四川省及各市州**政府公开**的事业单位公开招聘岗位一览表（原始 `.xlsx` 一并保留，可溯源）
- **本仓库不包含任何招聘平台的 JD 原文或截图**

## 目录

| 文件 | 作用 |
|---|---|
| `rag_intent.py` | 意图解析器（13 维度扫描 + 别名归一化 + 派生抑制） |
| `rag_core.py` | 检索入口（选库白名单 + 句柄缓存 + 执行期拒答 + count 分支） |
| `main.py` | FastAPI 服务（启动预热 + 响应 schema） |
| `csv_to_docs.py` | 数据重建（结构化 content + metadata → 双 collection） |
| `export_domains.py` | 值域导出（机关 / 单位 / 变体 / 城市别名 / 专业 / 学院） |
| `rag_eval.py` | 评测脚本（`--check` / `--parse-check` / `--parse-freeze` / `--smoke` / `--audit`） |
| `eval_result_*.md` | 三阶段检索评测留档 |
