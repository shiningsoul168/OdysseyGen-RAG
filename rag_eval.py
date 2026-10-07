# -*- coding: utf-8 -*-
"""
rag_eval.py — OdysseyGen RAG 评测脚本（替代 P@3）

为什么写它：
  P@3 的前提是"每个 query 只有 1 条正确答案"，你的库里多数 query 只有 1~3 条相关，
  所以 P@3 的上限被锁死在 0.33，分辨不出好坏。本脚本改成：
    - 排序类（实体/复合）→ Hit@1 / MRR@5 / Recall@K
    - 过滤类（字段）      → Purity@K（Top-K 里真满足字段条件的比例）+ gold 集大小
    - 拒答类（该空）      → Top-1 距离 + 分类型阈值判定（不做命中率）
  gold 集合一律**从 CSV 直接算**，不依赖检索引擎 —— 指标外部锚定，才算证据。

用法（在 F:\\OdysseyGen-RAG 下）：
    uv run python rag_eval.py --tag v0-baseline          # 重建前，纯向量基线
    uv run python rag_eval.py --tag v1-cosine            # 重建（cosine）后
    uv run python rag_eval.py --tag v2-filtered --mode filtered   # P2 意图分流后
    uv run python rag_eval.py --check                    # 不调 embedding：只核对 metadata 过滤条数

输出：终端表格 + eval_result_<tag>.md

你需要自己定的两件事（脚本里已给出推荐值，改这里就行）：
    THRESHOLD  —— 分类型拒答阈值（先用标定表看分布，再定）
    CASE 列表里的 intent —— P2 意图分流的目标形态，键名必须和你重建时落的 metadata 键一致

修订记录：
  v0.1  修正 3 条 case 的设计错误（K-E1 意图不纯；G-C5/G-C6 gold 为 0）
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
KY_CSV = os.path.join(HERE, "data", "kaoyan_data.csv")
KG_CSV = os.path.join(HERE, "data", "kaogong_data.csv")

# ---------------------------------------------------------------------------
# 本机配置
#   默认值 = 2026-09-27 实测的真实 collection 名（一个 DB_DIR，两个 collection）
#   如果同目录存在 rag_eval_config.py，则用它覆盖 —— 这样脚本本体升级不会冲掉你的配置
#   （吃过一次亏：覆盖脚本时把已填好的 COLLECTIONS 冲回 None，结果全部过滤=0）
# ---------------------------------------------------------------------------
COLLECTIONS = {"kaoyan": "kaoyan", "kaogong": "kaogong"}
THRESHOLD = {"kaoyan": None, "kaogong": None}     # 分类型拒答阈值；L2/cosine 不可跨空间比较

try:
    from rag_eval_config import COLLECTIONS as _C, THRESHOLD as _T
    COLLECTIONS, THRESHOLD = _C, _T
    print("[info] 已加载 rag_eval_config.py")
except ImportError:
    pass                                          # 没有配置文件就用上面的默认值
except Exception as _e:  # noqa
    print("[warn] rag_eval_config.py 有错，已回退默认配置：%s" % _e)

if any(v is None for v in COLLECTIONS.values()):
    print("[warn] COLLECTIONS 里有 None —— 会去查默认 collection（多半是空库），结果会全是 0")

DEFAULT_K = 5


# ============================================================
# 0. 数据
# ============================================================
def _load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


KY_ROWS = _load(KY_CSV)
KG_ROWS = _load(KG_CSV)


def g(row, key):
    return (row.get(key) or "").strip()


def ky_key_row(r):
    return "|".join([g(r, "学校"), g(r, "学院"), g(r, "专业"), g(r, "代码"), g(r, "年份")])


def ky_key_meta(m):
    return "|".join([(m.get(k) or "").strip() for k in ("学校", "学院", "专业", "代码", "年份")])


CODE_RE = re.compile(r"职位代码\s*[（(]?\s*(\d{6,})")


def kg_key_doc(doc):
    c = ((doc.metadata or {}).get("职位代码") or "").strip()
    if c:
        return c
    mt = CODE_RE.search(doc.page_content or "")
    return mt.group(1) if mt else ""


def row_key(row, dt):
    return ky_key_row(row) if dt == "kaoyan" else g(row, "职位代码")


def doc_key(doc, dt):
    if dt == "kaoyan":
        return ky_key_meta(doc.metadata or {})
    return kg_key_doc(doc)


def gold_keys(case):
    """gold 集合从 CSV 直接算 —— 不经过检索，这是指标的锚"""
    rows = KY_ROWS if case["dt"] == "kaoyan" else KG_ROWS
    return {row_key(r, case["dt"]) for r in rows if case["gold"](r)}


# ============================================================
# 1. 评测集（40 条：考研 16 + 考公 24；字段/实体/复合/该空 各 6~10 条）
#
#    intent 是"P2 意图分流的目标形态"：
#      ("字段名", "eq"|"in"|"lte"|"gte", 值)  → metadata where 过滤
#      ("__contains__", "contains", "字符串")  → where_document 精确包含
#    键名必须和你重建时实际写的 metadata 键一致，不一致就改这里。
# ============================================================
def _edu_level(v):
    """学历门槛数值化：中专1 大专2 本科3 研究生4 博士5；空/未识别→0"""
    if "博士" in v:
        return 5
    if "研究生" in v:
        return 4
    if "本科" in v:
        return 3
    if "大专" in v:
        return 2
    if "中专" in v:
        return 1
    return 0


CASE = [
    # ---------------- 考研：字段类（4） ----------------
    dict(id="K-F1", dt="kaoyan", cat="字段", q="川大计算机学院有哪些专业？",
         gold=lambda r: g(r, "学院") == "计算机学院",
         intent=[("学院", "eq", "计算机学院")],
         note="10 条（5 个专业 × 2 年），gold 大于 K，看纯度不看命中率"),
    dict(id="K-F2", dt="kaoyan", cat="字段", q="四川大学 2026 年有哪些专业公布了考试科目？",
         gold=lambda r: g(r, "年份") == "2026" and g(r, "考试科目") != "",
         intent=[("字段类型", "in", ("招生目录", "分数线+招生目录"))],
         note="A类9+C类11=20 条：过滤结果条数应等于 gold 条数"),
    dict(id="K-F3", dt="kaoyan", cat="字段", q="四川大学 2025 年有哪些专业有报录数据？",
         gold=lambda r: g(r, "年份") == "2025" and g(r, "统考报考") != "",
         intent=[("字段类型", "eq", "报录")],
         note="D类15 条"),
    dict(id="K-F4", dt="kaoyan", cat="字段", q="川大有哪些学硕专业？",
         gold=lambda r: g(r, "学位类型") == "学硕",
         intent=[("学位类型", "eq", "学硕")],
         note="跨年份，gold 较大"),

    # ---------------- 考研：实体类（5） ----------------
    #  ★ P2 设计结论：考研的「代码」在 metadata 里 → 实体类走 where 精确匹配，不走全文 contains
    #    实测 contains "085405" 会多召回 1 条（085400 的备注里提到了 085405），where 代码=085405 正好 2 条
    dict(id="K-E1", dt="kaoyan", cat="实体", q="专业代码 085405 是什么专业？",
         gold=lambda r: g(r, "代码") == "085405",
         intent=[("代码", "eq", "085405")],
         note="gold=2（2026 分数线 + 2025 报录）。仍会失败：085400 排在 085405 前面——"
              "五位数代码的末位差别向量区分不了，必须走 where"),
    dict(id="K-E2", dt="kaoyan", cat="实体", q="四川大学计算机科学与技术 2026 年复试分数线是多少？",
         gold=lambda r: g(r, "代码") == "081200" and g(r, "年份") == "2026" and g(r, "总分") != "",
         intent=[("代码", "eq", "081200"), ("年份", "eq", "2026"),
                 ("字段类型", "in", ("分数线", "分数线+招生目录"))],
         note="1 条(328)。★原始意图是『实体锚点 + 年份 + 字段』三条件叠加 —— "
              "P2 不能只做字段/实体/复合三选一，where 组装必须支持多条件 $and"),
    dict(id="K-E3", dt="kaoyan", cat="实体", q="川大计科考研考什么科目？",
         gold=lambda r: g(r, "代码") == "081200" and g(r, "考试科目") != "",
         intent=[("代码", "eq", "081200"), ("字段类型", "in", ("招生目录", "分数线+招生目录"))],
         note="1 条。v0 实测：2025 报录行排第 1，正确的 2026 行掉到第 4 → 『答对了但排在后面』，"
              "P@3 时代看不出来"),
    dict(id="K-E4", dt="kaoyan", cat="实体", q="川大计算机科学与技术报录比多少？",
         gold=lambda r: g(r, "代码") == "081200" and g(r, "统考报考") != "",
         intent=[("代码", "eq", "081200"), ("字段类型", "eq", "报录")],
         note="1 条，且只存在于 2025 —— 与 K-N2 配对看"),
    dict(id="K-E5", dt="kaoyan", cat="实体", q="川大软件工程专硕 2026 年分数线",
         gold=lambda r: g(r, "代码") == "085405" and g(r, "年份") == "2026" and g(r, "总分") != "",
         intent=[("代码", "eq", "085405"), ("年份", "eq", "2026"),
                 ("字段类型", "in", ("分数线", "分数线+招生目录"))],
         note="1 条(264)。易与学硕 083500(305) 混，考区分能力"),

    # ---------------- 考研：复合类（4） ----------------
    dict(id="K-C1", dt="kaoyan", cat="复合", q="川大计算机学院学硕的复试分数线",
         gold=lambda r: g(r, "学院") == "计算机学院" and g(r, "学位类型") == "学硕"
                        and g(r, "年份") == "2026" and g(r, "总分") != "",
         intent=[("学院", "eq", "计算机学院"), ("学位类型", "eq", "学硕"),
                 ("字段类型", "in", ("分数线", "分数线+招生目录"))],
         note="3 条：081200/083500/140500"),
    dict(id="K-C2", dt="kaoyan", cat="复合", q="川大计算机学院专硕的分数线",
         gold=lambda r: g(r, "学院") == "计算机学院" and g(r, "学位类型") == "专硕"
                        and g(r, "年份") == "2026" and g(r, "总分") != "",
         intent=[("学院", "eq", "计算机学院"), ("学位类型", "eq", "专硕"),
                 ("字段类型", "in", ("分数线", "分数线+招生目录"))],
         note="2 条：085400(360)/085405(264)"),
    dict(id="K-C3", dt="kaoyan", cat="复合", q="川大商学院 2026 年考什么科目？",
         gold=lambda r: g(r, "学院") == "商学院" and g(r, "年份") == "2026" and g(r, "考试科目") != "",
         intent=[("学院", "eq", "商学院"), ("字段类型", "in", ("招生目录", "分数线+招生目录"))],
         note="C类 4 条"),
    dict(id="K-C4", dt="kaoyan", cat="复合", q="川大经济学院 2026 年学硕考什么",
         gold=lambda r: g(r, "学院") == "经济学院" and g(r, "学位类型") == "学硕"
                        and g(r, "年份") == "2026" and g(r, "考试科目") != "",
         intent=[("学院", "eq", "经济学院"), ("学位类型", "eq", "学硕"),
                 ("字段类型", "in", ("招生目录", "分数线+招生目录"))],
         note="2 条"),

    # ---------------- 考研：该空类（3） ----------------
    dict(id="K-N1", dt="kaoyan", cat="该空", q="西南财经大学金融专硕 2026 分数线",
         gold=lambda r: False, intent=[("__contains__", "contains", "西南财经大学")],
         note="库里只有川大。正确行为：返回'暂无数据'，不是返回川大金融"),
    dict(id="K-N2", dt="kaoyan", cat="该空", q="四川大学计算机科学与技术 2026 年报录比",
         gold=lambda r: False,
         # ★意图已修正（2026-09-28）：原来只标 contains "081200" → 命中 2 条（2025+2026）→ 非空 → 不拒答。
         #   本用例的本质是"K-E4 的年份前提不成立"，所以意图 = K-E4 + 年份=2026 → 0 条
         intent=[("代码", "eq", "081200"), ("字段类型", "eq", "报录"), ("年份", "eq", "2026")],
         note="★重要：081200 只有 2025 报录。正确行为是标注'库内为2025数据'或拒答，不能拿旧数据当新数据"),
    dict(id="K-N3", dt="kaoyan", cat="该空", q="电子科技大学 2026 招生目录",
         gold=lambda r: False, intent=[("__contains__", "contains", "电子科技大学")],
         note="库里只有川大"),

    # ---------------- 考公：字段类（6） ----------------
    dict(id="G-F1", dt="kaogong", cat="字段", q="有哪些岗位要求中共党员？",
         gold=lambda r: g(r, "政治面貌") != "",
         intent=[("政治面貌", "eq", "中共党员")],
         note="180 条（151含预备+29）→ 归一化后都是'中共党员'"),
    dict(id="G-F2", dt="kaogong", cat="字段", q="要求 2 年以上基层工作经历的岗位有哪些？",
         gold=lambda r: g(r, "基层年限") == "2年以上基层工作经历",
         intent=[("基层要求", "eq", "2年以上")],
         note="234 条。与 G-F5 的 240 条配对，验证'2年以上'和'有要求'必须分开"),
    dict(id="G-F3", dt="kaogong", cat="字段", q="大专学历可以报考的岗位有哪些？",
         gold=lambda r: (g(r, "学历要求").find("大专") >= 0) or (g(r, "学历要求").find("中专") >= 0),
         intent=[("学历门槛", "lte", 2)],
         note="734 条。★这条证明'学历要求'只归一化成字符串不够，必须落数值门槛"),
    dict(id="G-F4", dt="kaogong", cat="字段", q="要求研究生学历的岗位有哪些？",
         gold=lambda r: g(r, "学历要求").find("研究生") >= 0,
         intent=[("学历门槛", "gte", 4)],
         note="576 条"),
    dict(id="G-F5", dt="kaogong", cat="字段", q="有基层工作经历要求的岗位有哪些？",
         gold=lambda r: g(r, "基层年限") != "",
         intent=[("基层要求", "in", ("2年以上", "有要求(年限未明)"))],
         note="240 条。与 G-F2 差 6 条：措辞模糊必须由判定规则裁决"),
    dict(id="G-F6", dt="kaogong", cat="字段", q="博士学历要求的岗位有哪些？",
         gold=lambda r: g(r, "学历要求").find("博士") >= 0,
         intent=[("学历门槛", "gte", 5)],
         note="18 条，gold 小、可判别"),

    # ---------------- 考公：实体类（6） ----------------
    dict(id="G-E1", dt="kaogong", cat="实体", q="广安市融媒体中心有哪些岗位？",
         gold=lambda r: g(r, "招聘单位") == "广安市融媒体中心",
         intent=[("__contains__", "contains", "广安市融媒体中心")],
         note="2 条。原问法『招全媒体记者吗』的 gold 只算 1 条，而 contains 单位名必然召回 2 条 —— "
              "是 gold 比 intent 窄，不是过召回。若真要测『实体+职位』双条件，"
              "需要 where_document 内嵌 $and（P2 待验证）"),
    dict(id="G-E2", dt="kaogong", cat="实体", q="南充职业技术学院有哪些岗位？",
         gold=lambda r: g(r, "招聘单位") == "南充职业技术学院",
         intent=[("__contains__", "contains", "南充职业技术学院")],
         note="2 条"),
    dict(id="G-E3", dt="kaogong", cat="实体", q="成都市农林科学院招什么岗位？",
         gold=lambda r: g(r, "招聘单位") == "成都市农林科学院",
         intent=[("__contains__", "contains", "成都市农林科学院")],
         note="2 条"),
    dict(id="G-E4", dt="kaogong", cat="实体", q="中共岳池县委党校的教师岗要求什么学历？",
         gold=lambda r: g(r, "招聘单位") == "中共岳池县委党校",
         intent=[("__contains__", "contains", "中共岳池县委党校")],
         note="1 条(214003079096)：研究生学历 + 党员"),
    dict(id="G-E5", dt="kaogong", cat="实体", q="邛崃市临邛社区卫生服务中心招哪些岗位？",
         gold=lambda r: g(r, "招聘单位") == "邛崃市临邛社区卫生服务中心",
         intent=[("__contains__", "contains", "邛崃市临邛社区卫生服务中心")],
         note="2 条。考长文档里的实体召回（备注含『工作地点为前进院区…』）"),
    dict(id="G-E6", dt="kaogong", cat="实体", q="四川省水利厅有哪些岗位？",
         gold=lambda r: g(r, "招录机关") == "省水利厅",
         intent=[("__contains__", "contains", "省水利厅")],
         note="★别名问题已用数据量化：contains『四川省水利厅』→ 1 条，contains『省水利厅』→ 123 条，"
              "而 gold=123 → **归一化后完全精确、零过召回**。所以别名规则是硬的："
              "实体串先去省份前缀（四川省→省）再 contains。不做归一化，这条会从 v0 的"
              "『命中 123 条』断崖掉到 1 条"),

    # ---------------- 考公：复合类（6） ----------------
    dict(id="G-C1", dt="kaogong", cat="复合", q="成都有哪些计算机相关的事业单位岗位？",
         gold=lambda r: g(r, "招录机关") == "成都市" and "计算机" in g(r, "专业要求"),
         intent=[("城市", "eq", "成都市"), ("__contains__", "contains", "计算机")],
         note="你原评测里 v4 从 0→1.0 的那条"),
    dict(id="G-C2", dt="kaogong", cat="复合", q="广安要求党员的岗位有哪些？",
         gold=lambda r: g(r, "招录机关") == "广安市" and g(r, "政治面貌") != "",
         intent=[("城市", "eq", "广安市"), ("政治面貌", "eq", "中共党员")],
         note="城市 + 字段。v0 实测纯度 0.000：召回的全是广安岗，但没一个是党员岗 → 城市语义压过字段"),
    dict(id="G-C3", dt="kaogong", cat="复合", q="凉山州大专可以报的岗位有哪些？",
         gold=lambda r: g(r, "招录机关") == "凉山州" and _edu_level(g(r, "学历要求")) <= 2
                        and _edu_level(g(r, "学历要求")) > 0,
         intent=[("城市", "eq", "凉山州"), ("学历门槛", "lte", 2)],
         note="城市 + 数值门槛"),
    dict(id="G-C4", dt="kaogong", cat="复合", q="成都市要求研究生学历的岗位有哪些？",
         gold=lambda r: g(r, "招录机关") == "成都市" and _edu_level(g(r, "学历要求")) >= 4,
         intent=[("城市", "eq", "成都市"), ("学历门槛", "gte", 4)],
         note="城市 + 数值门槛"),
    dict(id="G-C5", dt="kaogong", cat="复合", q="乐山市有基层工作经历要求的岗位",
         gold=lambda r: g(r, "招录机关") == "乐山市" and g(r, "基层年限") != "",
         intent=[("城市", "eq", "乐山市"), ("基层要求", "in", ("2年以上", "有要求(年限未明)"))],
         note="城市 + 字段，gold=56。（原写法用南充市，实测该市基层要求 0 条 → 已换乐山市）"),
    dict(id="G-C6", dt="kaogong", cat="复合", q="本科学历的党务管理岗有哪些？",
         gold=lambda r: "党务管理" in g(r, "职位名称")
                        and 0 < _edu_level(g(r, "学历要求")) <= 3,
         intent=[("学历门槛", "lte", 3), ("__contains__", "contains", "党务管理")],
         note="gold=3（省农科院 + 国家粮食和物资储备局四川局×2）。"
              "原写法带'成都市'实测 gold=0 —— 成都没有党务管理岗，已去城市条件"),

    # ---------------- 考公：该空类（6） ----------------
    dict(id="G-N1", dt="kaogong", cat="该空", q="四川旅游学院的岗位有哪些？",
         gold=lambda r: False, intent=[("__contains__", "contains", "四川旅游学院")],
         note="0 条 → 应返回'暂无数据'。v0 Top1 距离 0.5995，和你文档里记的数字一致（可复现）"),
    dict(id="G-N2", dt="kaogong", cat="该空", q="深圳有哪些事业单位岗位？",
         gold=lambda r: False, intent=[("__contains__", "contains", "深圳")],
         note="0 条"),
    dict(id="G-N3", dt="kaogong", cat="该空", q="成都有哪些公务员岗位？",
         gold=lambda r: False,
         # ★意图已修正（2026-09-28）：原来只标 contains "公务员" → 正文里有文档提到"公务员" → 非空 → 不拒答。
         #   本用例本质是"口径不匹配"：考试类型只有 市州事业单位/省属事业单位/省属中小学教师，没有"公务员"
         intent=[("城市", "eq", "成都市"), ("考试类型", "eq", "公务员")],
         note="★诚实性用例：库内是'事业单位'招聘，不是公务员。正确行为是说明数据口径，不是硬凑岗位"),
    dict(id="G-N4", dt="kaogong", cat="该空", q="2027 年四川省事业单位岗位有哪些？",
         gold=lambda r: False, intent=[("年份", "eq", "2027")],
         note="库内只有 2026"),
    dict(id="G-N5", dt="kaogong", cat="该空", q="广安市融媒体中心招数据科学家吗？",
         gold=lambda r: False, intent=[("__contains__", "contains", "数据科学家")],
         note="★单位存在、岗位不存在。正确行为：列出该单位真实岗位并说明无此岗。"
              "v0 Top1 距离 0.4890 —— 是考公该空组里**最近**的一条，却是最该拒答的一条"),
    dict(id="G-N6", dt="kaogong", cat="该空", q="四川省考省级机关的岗位有哪些？",
         gold=lambda r: False, intent=[("__contains__", "contains", "省级机关")],
         note="库内为市州/省属事业单位公开招聘，无'省级机关'口径"),
]

CATS = ["字段", "实体", "复合", "该空"]


# ============================================================
# 2. 检索
# ============================================================
_DB_CACHE = {}


def _db(dt):
    """按 data_type 取 collection。

    ⚠ 口径必须和 rag_core.search() 一致：
       同一个 DB_DIR（一个库），靠 collection_name 区分 kaoyan / kaogong。
       如果这里和 rag_core 选的库不一样，评测数字就不能代表线上行为。
       （rag_core 已改成按 collection_name 建 handle；这里复用它的 DB_DIR 和 embeddings）
    """
    if dt not in _DB_CACHE:
        from rag_core import DB_DIR, embeddings
        from langchain_chroma import Chroma
        name = COLLECTIONS.get(dt) or dt
        _DB_CACHE[dt] = Chroma(persist_directory=DB_DIR,
                               embedding_function=embeddings,
                               collection_name=name)
    return _DB_CACHE[dt]


def _where(intent):
    """组装 Chroma where。

    ⚠ 关键：新版 Chroma 要求 where **顶层只能有一个算子键**。
      多条件必须显式包成 {"$and": [ {...}, {...} ]}，
      否则报 "Expected where to have exactly one operator"。
      —— P2 你自己的检索实现（rag_core / 意图分流）也必须这么写。
    """
    parts = []
    for f, op, val in intent:
        if f == "__contains__":
            continue
        if op == "eq":
            parts.append({f: val})
        elif op == "in":
            parts.append({f: {"$in": list(val)}})
        elif op == "lte":
            parts.append({f: {"$lte": val}})
        elif op == "gte":
            parts.append({f: {"$gte": val}})
        else:
            raise ValueError("不支持的算子: %s" % op)
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _and(out, extra):
    """把一个过滤条件并进已有的 where（保持"顶层单算子"约束）"""
    if not out:
        return extra
    if "$and" in out:
        return {"$and": out["$and"] + [extra]}
    return {"$and": [out, extra]}


def _contains(intent):
    for f, op, val in intent:
        if f == "__contains__":
            return val
    return None


def retrieve(query, dt, k, case, mode):
    """返回 [(doc, distance), ...]"""
    db = _db(dt)
    if mode == "raw":
        return db.similarity_search_with_score(query, k=k, filter={"data_type": dt})

    # ---- filtered：P2 目标形态（参考实现，用到私有 API，按你的 langchain-chroma 版本微调）----
    where = _where(case["intent"])
    if COLLECTIONS.get(dt) is None:
        where = _and(where, {"data_type": dt})   # 不能直接 where["data_type"]=...（多键会破坏单算子约束）
    contains = _contains(case["intent"])
    emb = db._embedding_function.embed_query(query)
    kwargs = {"query_embeddings": [emb], "n_results": k, "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    if contains:
        kwargs["where_document"] = {"$contains": contains}
    res = db._collection.query(**kwargs)

    from langchain_core.documents import Document
    docs, dists = [], []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        docs.append(Document(page_content=text, metadata=meta or {}))
        dists.append(dist)
    return list(zip(docs, dists))


def space_of(dt):
    try:
        meta = _db(dt)._collection.metadata or {}
        return meta.get("hnsw:space", "l2(默认)")
    except Exception as e:  # noqa
        return "未知(%s)" % e


# ============================================================
# 3. 打分
# ============================================================
def score_case(case, mode, k):
    gk = gold_keys(case)
    out = dict(case=case, gold_size=len(gk), k=k, err=None, top1_dist=None, top1_preview="",
               hit1=None, mrr=None, recall=None, purity=None, retrieved=[])
    try:
        rows = retrieve(case["q"], case["dt"], k, case, mode)
    except Exception as e:  # noqa
        out["err"] = "%s: %s" % (type(e).__name__, e)
        return out

    keys, dists, previews = [], [], []
    for doc, dist in rows:
        keys.append(doc_key(doc, case["dt"]))
        dists.append(float(dist))
        previews.append((doc.page_content or "")[:60].replace("\n", " "))
    out["retrieved"] = list(zip(keys, dists, previews))
    if dists:
        out["top1_dist"] = dists[0]
        out["top1_preview"] = previews[0]

    if case["cat"] == "该空" or not gk:
        return out

    hits = [i for i, key in enumerate(keys) if key in gk]
    out["hit1"] = 1.0 if 0 in hits else 0.0
    out["mrr"] = 1.0 / (hits[0] + 1) if hits else 0.0
    out["recall"] = len(set(hits)) / float(len(gk))
    out["purity"] = len(hits) / float(len(keys)) if keys else 0.0
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, nd=3):
    return "-" if x is None else ("%.*f" % (nd, x))


def dist_line(vals):
    if not vals:
        return "-"
    vals = sorted(vals)
    return "n=%d min=%.4f 中位=%.4f max=%.4f" % (
        len(vals), vals[0], statistics.median(vals), vals[-1])


# ============================================================
# 4. --check：不调 embedding，只核对"过滤条数 == gold 条数"
#    重建完先跑这个，能在花一分钱 embedding 之前发现 metadata 落错
# ============================================================
def run_check():
    print("=" * 78)
    print("metadata 核对（不调用 embedding）：过滤结果条数 应等于 gold 条数")
    print("=" * 78)
    bad = 0
    for case in CASE:
        if case["cat"] == "该空":
            continue
        gk = gold_keys(case)
        where = _where(case["intent"])
        contains = _contains(case["intent"])
        if not where and not contains:
            print("[跳过] %-6s %s  （无过滤条件）" % (case["id"], case["q"]))
            continue
        if COLLECTIONS.get(case["dt"]) is None:
            where = _and(where, {"data_type": case["dt"]})
        try:
            kwargs = {"include": []}
            if where:
                kwargs["where"] = where
            if contains:
                kwargs["where_document"] = {"$contains": contains}
            got = _db(case["dt"])._collection.get(**kwargs)
            n = len(got.get("ids") or [])
        except Exception as e:  # noqa
            print("[错误] %-6s %s -> %s" % (case["id"], case["q"], e))
            bad += 1
            continue
        flag = "OK " if n == len(gk) else "差异"
        if n != len(gk):
            bad += 1
        mark = "+contains" if contains else ""
        print("[%s] %-6s gold=%-5d 过滤=%-5d %-10s %s" % (flag, case["id"], len(gk), n, mark, case["q"]))
    print("-" * 78)
    print("不一致 %d 条。注意：gold 用的是 CSV 原始字段，过滤用的是重建后归一化的 metadata；" % bad)
    print("       两者口径差异（如空值→'不限'/'否'）允许存在，但必须能逐条解释清楚。")
    print("       ⚠ where_document 是**全文包含**、不是字段包含：若某条 +contains 的过滤数 > gold，")
    print("         说明该关键词也出现在别的字段里（过召回），由后续向量排序+阈值收，或改 gold 定义。")


# ============================================================
# 4b. 意图解析测试（--parse-check / --parse-freeze / --smoke / --audit）
#     解析层：断言 parse_intent 的 where / contains / unsupported / strategy
#     执行层：collection.get(...) 计命中条数 → 0 条升级 reject；占比 >50% 出 warning
#     全程零 embedding、秒级
# ============================================================

TOTAL_HINT = {"kaoyan": 69, "kaogong": 5649}   # 占比判据的分母兜底


def norm_struct(node):
    """递归规范化后比较（**不比字符串**）。规则表 §12.5

    - 顶层多键 → 视作 $and（Chroma 顶层只允许一个键）
    - $and / $or 列表排序；**单项塌缩成裸条件**（Chroma 硬约束：$and 不能只有 1 项）
    - $in / $nin 列表排序
    - 空 → None
    """
    if not isinstance(node, dict) or not node:
        return None
    if len(node) > 1:
        node = {"$and": [{k: v} for k, v in node.items()]}
    ((k, v),) = node.items()
    if k in ("$and", "$or"):
        items = [norm_struct(x) for x in v]
        items = [x for x in items if x is not None]
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        items.sort(key=lambda d: json.dumps(d, sort_keys=True, ensure_ascii=False))
        return {k: items}
    if isinstance(v, dict):
        out = {}
        for op, val in v.items():
            out[op] = sorted(val, key=str) if (op in ("$in", "$nin") and isinstance(val, list)) else val
        return {k: out}
    return {k: v}


def norm_contains(terms, op="or"):
    """contains 词表 → where_document 结构（§12.6：多实体用 $or）"""
    if not terms:
        return None
    terms = sorted(terms)
    if len(terms) == 1:
        return {"$contains": terms[0]}
    if op == "and":
        return {"$and": [{"$contains": t} for t in terms]}
    return {"$or": [{"$contains": t} for t in terms]}


def _total(dt):
    try:
        return _db(dt)._collection.count()
    except Exception:  # noqa
        return TOTAL_HINT.get(dt)


def _count_rows(dt, where, contains, contains_op="or"):
    """执行层：过滤命中条数（零 embedding）。无条件 → None"""
    if not where and not contains:
        return None
    kwargs = {"include": []}
    if where:
        kwargs["where"] = where
    wd = norm_contains(contains, contains_op)
    if wd:
        kwargs["where_document"] = wd
    return len(_db(dt)._collection.get(**kwargs).get("ids") or [])


def finalize(strategy, rows):
    """执行期升级：filtered 但 0 条 → reject（规则表 §2）"""
    if strategy in ("filtered", "reject") and rows == 0:
        return "reject"
    return strategy


def too_broad(rows, dt):
    """warning 判据：命中占比 > 50%（规则表 §6.5）"""
    if rows is None:
        return False
    return rows / float(_total(dt) or 1) > 0.5


# ------------------------------------------------------------
# 用例表：#1~#27 来自规则表 §7；#28 覆盖"unsupported 不阻断 filtered"
#   where/contains/unsupported/final 为**冻结期望**；rows/warn 由执行期核对
#   ⚠ 标 ★ 的是首跑需要用 --parse-freeze 核对的（我不确定你的解析器是否产出该键）
# ------------------------------------------------------------
PARSE_CASES = [
    dict(id=1, q="不限政治面貌的岗位有哪些？", dt="kaogong",
         where={"政治面貌": "不限"}, rows=5469, warn=True, final="filtered"),
    dict(id=2, q="不需要基层工作经历的岗位", dt="kaogong",
         where={"基层要求": "否"}, rows=5409, warn=True, final="filtered"),
    dict(id=3, q="川大有哪些计算机专业？", dt="kaoyan",   # 忠实解析：走 scan_major → contains 计算机
         where={"学校": "四川大学"}, contains=["计算机"],
         rows=None, warn=False, final="filtered"),          # rows 跑 --parse-freeze 后填
    dict(id=4, q="电子科大 2026 分数线", dt="kaoyan",          # ★首跑核对（年份是否产出）
         where={"$and": [{"学校": "电子科技大学"}, {"年份": "2026"},
                         {"字段类型": {"$in": ["分数线", "分数线+招生目录"]}}]},
         rows=0, warn=False, final="reject"),
    dict(id=5, q="今年有哪些岗位？", dt="kaogong",
         where={"年份": "2026"}, rows=5649, warn=True, final="filtered"),
    dict(id=6, q="成都或德阳的党员岗", dt="kaogong",
         where={"$and": [{"城市": {"$in": ["成都市", "德阳市"]}}, {"政治面貌": "中共党员"}]},
         rows=23, warn=False, final="filtered"),
    dict(id=7, q="招 3 人以上的岗位", dt="kaogong",
         where=None, unsupported=["招录人数≥3"], strategy="reject", final="reject"),
    dict(id=8, q="有哪些岗位适合应届生？", dt="kaogong",
         where={"基层要求": "否"}, rows=5409, warn=True, final="filtered"),
    dict(id=9, q="广安要求党员的岗位", dt="kaogong",
         where={"$and": [{"城市": "广安市"}, {"政治面貌": "中共党员"}]}, rows=5, warn=False, final="filtered"),
    dict(id=10, q="成都有哪些计算机相关的岗位", dt="kaogong",
         where={"城市": "成都市"}, contains=["计算机"], rows=25, warn=False, final="filtered"),
    dict(id=11, q="大专学历可以报考的岗位", dt="kaogong",
         where={"学历门槛": {"$lte": 2}}, rows=734, warn=False, final="filtered"),
    dict(id=12, q="要求研究生学历的岗位", dt="kaogong",
         where={"学历门槛": {"$in": [4, 5]}}, rows=576, warn=False, final="filtered"),
    dict(id=13, q="四川省水利厅有哪些岗位", dt="kaogong",
         where={"招录机关": "省水利厅"}, rows=123, warn=False, final="filtered"),
    dict(id=14, q="川大计科考什么科目", dt="kaoyan",
         where={"$and": [{"学校": "四川大学"}, {"代码": "081200"},
                         {"字段类型": {"$in": ["招生目录", "分数线+招生目录"]}}]},
         rows=1, warn=False, final="filtered"),
    dict(id=15, q="川大计科报录比", dt="kaoyan",
         where={"$and": [{"学校": "四川大学"}, {"代码": "081200"}, {"字段类型": "报录"}]},
         rows=1, warn=False, final="filtered"),
    dict(id=16, q="除了成都的岗位", dt="kaogong",
         where={"$and": [{"城市": {"$ne": "成都市"}},
                         {"城市": {"$nin": ["省属", "中央驻川", "国企"]}}]},
         rows=4641, warn=True, final="filtered"),
    dict(id=17, q="本科可报的党员岗", dt="kaogong",
         where={"$and": [{"学历门槛": {"$lte": 3}}, {"政治面貌": "中共党员"}]},
         rows=120, warn=False, final="filtered"),
    dict(id=18, q="凉山州大专可以报的岗位", dt="kaogong",
         where={"$and": [{"城市": "凉山州"}, {"学历门槛": {"$lte": 2}}]}, rows=32, warn=False, final="filtered"),
    dict(id=19, q="四川旅游学院的岗位", dt="kaogong",
         contains=["四川旅游学院"], rows=0, warn=False, final="reject"),
    dict(id=20, q="2027 年四川省事业单位岗位", dt="kaogong",
         where={"年份": "2027"}, rows=0, warn=False, final="reject"),
    dict(id=21, q="川大软工专硕分数线", dt="kaoyan",       # 跑通 = 派生抑制生效
         where={"$and": [{"学校": "四川大学"}, {"代码": "085405"},
                         {"字段类型": {"$in": ["分数线", "分数线+招生目录"]}}]},
         rows=1, warn=False, final="filtered"),
    dict(id=22, q="085405 是什么专业？", dt="kaoyan",
         where={"代码": "085405"}, rows=2, warn=False, final="filtered"),
    dict(id=23, q="非全日制可以报吗", dt="kaogong",
         where=None, unsupported=["学习方式"], strategy="reject", final="reject"),
    dict(id=24, q="全日制研究生岗位", dt="kaoyan",          # ★首跑核对（unsupported 文案）
         where=None, unsupported=["学习方式=全日制"], strategy="reject", final="reject"),
    dict(id=25, q="广安有哪些岗位", dt="kaogong",
         where={"城市": "广安市"}, rows=268, warn=False, final="filtered"),
    dict(id=26, q="水利厅的岗位", dt="kaogong",
         where={"招录机关": "省水利厅"}, rows=123, warn=False, final="filtered"),
    dict(id=27, q="成都市农林科学院和南充职业技术学院有哪些岗位", dt="kaogong",
         # 单位名走 metadata 精确（§4.6 顺序）；scan_city 排除单位名内部 → 不再产出城市条件
         where={"招聘单位": {"$in": ["成都市农林科学院", "南充职业技术学院"]}},
         rows=4, warn=False, final="filtered"),
    dict(id=29, q="四川旅游学院和成都工业学院有哪些岗位", dt="kaogong",
         # 真正测 where_document $or：两个实体库里都没有 → 落 pass 2 → $or → 0 条 → reject
         contains=["四川旅游学院", "成都工业学院"], contains_op="or",
         rows=0, warn=False, final="reject"),
    dict(id=28, q="全日制本科能报吗", dt="kaogong",         # unsupported 不阻断 filtered
         where={"学历门槛": {"$lte": 3}}, unsupported=["学习方式"],
         rows=5068, warn=True, strategy="filtered", final="filtered"),
    dict(id=30, q="四川大学 软件工程 考研 分数线 考试科目", dt="kaoyan",
         # ★ 真实 query 回归（Java buildQuery 造出来的原话）。
         #   起因：29 条解析回归 + 40 条评测集全过，但没人测过"Java 真实构造的 query"，
         #   结果"软件工程"被静默丢弃 → count=43 而不是 2。
         #   → 教训：真实 query 要单独当一类回归用例。
         where={"$and": [{"学校": "四川大学"}, {"专业": "软件工程"},
                         {"字段类型": {"$in": ["分数线", "分数线+招生目录"]}}]},
         rows=2, warn=False, final="filtered"),
]


def run_parse_check():
    try:
        from rag_intent import parse_intent
    except Exception as e:  # noqa
        print("[错误] 无法导入 rag_intent.parse_intent：%s" % e)
        return 1

    print("=" * 78)
    print("意图解析回归  %d 条 ｜ 解析层 + 执行层 ｜ 零 embedding" % len(PARSE_CASES))
    print("=" * 78)

    bad = 0
    for c in PARSE_CASES:
        problems = []
        r = None
        try:
            r = parse_intent(c["q"], c["dt"])
        except Exception as e:  # noqa
            problems.append("parse_intent 抛异常: %s" % e)

        rows = None
        if r is not None:
            gw, ew = norm_struct(r.get("where")), norm_struct(c.get("where"))
            if gw != ew:
                problems.append("where 期望 %s ｜ 实际 %s"
                                % (json.dumps(ew, ensure_ascii=False, sort_keys=True),
                                   json.dumps(gw, ensure_ascii=False, sort_keys=True)))
            op = c.get("contains_op", "or")
            gc, ec = norm_contains(r.get("contains"), op), norm_contains(c.get("contains"), op)
            if gc != ec:
                problems.append("contains 期望 %s ｜ 实际 %s" % (ec, gc))
            gu, eu = sorted(r.get("unsupported") or []), sorted(c.get("unsupported") or [])
            if gu != eu:
                problems.append("unsupported 期望 %s ｜ 实际 %s" % (eu, gu))
            if "strategy" in c and r.get("strategy") != c["strategy"]:
                problems.append("strategy(解析期) 期望 %s ｜ 实际 %s" % (c["strategy"], r.get("strategy")))
            try:
                rows = _count_rows(c["dt"], r.get("where"), r.get("contains"), op)
            except Exception as e:  # noqa
                problems.append("执行期查询失败: %s" % e)
            if rows is not None and c.get("rows") is not None and rows != c["rows"]:
                problems.append("命中条数 期望 %s ｜ 实际 %s" % (c["rows"], rows))
            if rows is not None and "warn" in c and too_broad(rows, c["dt"]) != c["warn"]:
                problems.append("warning 期望 %s ｜ 实际 %s（%s/%s，占比 %.0f%%）"
                                % (c["warn"], too_broad(rows, c["dt"]), rows, _total(c["dt"]),
                                   100.0 * rows / (_total(c["dt"]) or 1)))
            if rows is not None and "final" in c and finalize(r.get("strategy"), rows) != c["final"]:
                problems.append("strategy(执行期) 期望 %s ｜ 实际 %s"
                                % (c["final"], finalize(r.get("strategy"), rows)))

        if problems:
            bad += 1
            print("[FAIL] #%-3d %s" % (c["id"], c["q"]))
            for p in problems:
                print("         → %s" % p)
        else:
            print("[OK  ] #%-3d %s  → %s ｜ %s 条" % (c["id"], c["q"], c["final"], rows))

    print("-" * 78)
    print("通过 %d / %d" % (len(PARSE_CASES) - bad, len(PARSE_CASES)))
    if bad:
        print("提示：若某条只是键多了/少了，先跑 `--parse-freeze` 看实际输出，")
        print("      人工确认**哪个对**再改 PARSE_CASES —— 不要把期望直接改成实际。")
    return bad


def run_parse_freeze():
    """把当前实际输出打印成可粘贴的用例行（人工核对后再贴回 PARSE_CASES）"""
    try:
        from rag_intent import parse_intent
    except Exception as e:  # noqa
        print("[错误] 无法导入 rag_intent.parse_intent：%s" % e)
        return
    print("# 由 --parse-freeze 生成。请逐条人工核对：期望是**规则表说的**，不是**解析器吐的**。")
    print("PARSE_CASES = [")
    for c in PARSE_CASES:
        r = parse_intent(c["q"], c["dt"])
        op = c.get("contains_op", "or")
        rows = _count_rows(c["dt"], r.get("where"), r.get("contains"), op)
        warn = too_broad(rows, c["dt"])
        fin = finalize(r.get("strategy"), rows) if rows is not None else r.get("strategy")
        print("    dict(id=%d, q=%r, dt=%r, where=%r, contains=%r, unsupported=%r," %
              (c["id"], c["q"], c["dt"], r.get("where"), r.get("contains"), r.get("unsupported")))
        print("         rows=%r, warn=%r, strategy=%r, final=%r),     # 实际" %
              (rows, warn, r.get("strategy"), fin))
    print("]")


def run_smoke():
    """§9 第 0 步：算子支持性 smoke test（零 embedding）。括号里是期望条数"""
    tests = [
        ("where $ne", "kaogong", {"城市": {"$ne": "成都市"}}, None, 5263),
        ("where $nin", "kaogong", {"城市": {"$nin": ["省属", "中央驻川", "国企"]}}, None, 5027),
        ("where $in + $and", "kaogong",
         {"$and": [{"城市": {"$in": ["成都市", "德阳市"]}}, {"政治面貌": "中共党员"}]}, None, 23),
        ("wd $contains", "kaogong", None, {"$contains": "四川旅游学院"}, 0),
        ("wd $or", "kaogong", None,
         {"$or": [{"$contains": "成都市农林科学院"}, {"$contains": "南充职业技术学院"}]}, 4),
        ("wd $regex", "kaogong", None, {"$regex": "214000001001"}, 1),
        ("where + wd 同用", "kaogong", {"城市": "成都市"}, {"$contains": "计算机"}, 25),
    ]
    print("=" * 78)
    print("算子 smoke test（不调 embedding）—— 源码已验证支持，这里验证 Rust 层行为一致")
    print("=" * 78)
    bad = 0
    for name, dt, where, wd, want in tests:
        kwargs = {"include": []}
        if where:
            kwargs["where"] = where
        if wd:
            kwargs["where_document"] = wd
        try:
            n = len(_db(dt)._collection.get(**kwargs).get("ids") or [])
            ok = (n == want) if want is not None else True
            bad += 0 if ok else 1
            print("[%s] %-18s → %d 条（期望 %s）" % ("OK  " if ok else "FAIL", name, n, want))
        except Exception as e:  # noqa
            bad += 1
            print("[FAIL] %-18s → %s: %s" % (name, type(e).__name__, e))
    print("-" * 78)
    print("失败 %d 项。任何一项失败 → 该算子不许进 parse_intent。" % bad)
    return bad


def run_audit():
    """metadata 取值审计：分布 + 可疑取值告警（规则表 §8）"""
    SUSPECT = ("人", "分", "%", "元", "年", "月", "；", "。")
    print("=" * 78)
    print("metadata 取值审计 —— 取值里出现 人/分/%%/元/年/月/；/。 的键要人工看")
    print("=" * 78)
    for dt in ("kaoyan", "kaogong"):
        try:
            metas = _db(dt)._collection.get(include=["metadatas"]).get("metadatas") or []
        except Exception as e:  # noqa
            print("[%s] 读取失败：%s" % (dt, e))
            continue
        keys = {}
        for m in metas:
            for k, v in (m or {}).items():
                keys.setdefault(k, Counter())[v] += 1
        print("\n### %s（%d 条）" % (dt, len(metas)))
        for k in sorted(keys):
            cnt = keys[k]
            sus = [str(v) for v in cnt if any(ch in str(v) for ch in SUSPECT)]
            print("  【%s】%d 个取值%s" % (k, len(cnt), "   ⚠ 人工看：%d 个可疑" % len(sus) if sus else ""))
            for v, n in cnt.most_common(6):
                print("      %-30s %d" % (v, n))
            if len(cnt) > 6:
                print("      …（还有 %d 个取值）" % (len(cnt) - 6))


# ============================================================
# 5. 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="结果文件后缀，如 v0-baseline")
    ap.add_argument("--mode", default="raw", choices=["raw", "filtered"])
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--show", action="store_true", help="打印每条 case 的 Top-K 明细")
    ap.add_argument("--check", action="store_true", help="只核对 metadata 过滤条数（检索侧）")
    ap.add_argument("--parse-check", action="store_true", help="意图解析回归：28 条，解析层+执行层")
    ap.add_argument("--parse-freeze", action="store_true", help="打印实际解析输出（人工核对后贴回）")
    ap.add_argument("--smoke", action="store_true", help="算子支持性 smoke test（§9 第 0 步）")
    ap.add_argument("--audit", action="store_true", help="metadata 取值审计")
    args = ap.parse_args()

    if args.smoke:
        sys.exit(1 if run_smoke() else 0)
    if args.audit:
        run_audit()
        return
    if args.parse_freeze:
        run_parse_freeze()
        return
    if args.parse_check:
        sys.exit(1 if run_parse_check() else 0)
    if args.check:
        run_check()
        return

    print("=" * 78)
    print("OdysseyGen RAG 评测  mode=%s  K=%d  tag=%s" % (args.mode, args.k, args.tag))
    print("距离空间：kaoyan=%s  kaogong=%s   ← L2 与 cosine 不可跨空间比较" %
          (space_of("kaoyan"), space_of("kaogong")))
    print("CSV：考研 %d 条 / 考公 %d 条   评测集 %d 条" % (len(KY_ROWS), len(KG_ROWS), len(CASE)))
    print("=" * 78)

    results = [score_case(c, args.mode, args.k) for c in CASE]

    lines = []
    lines.append("| id | 类 | gold | Hit@1 | MRR | Recall@%d | 纯度 | Top1距离 | 结果 |" % args.k)
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r["case"]
        if r["err"]:
            verdict = "错误"
        elif c["cat"] == "该空":
            if r["top1_dist"] is None:
                # 过滤结果为空 → 直接拒答（P2 的新机制，不再依赖距离阈值）
                verdict = "正确拒答(过滤为空)"
            else:
                th = THRESHOLD.get(c["dt"])
                if th is None:
                    verdict = "待定阈值"
                else:
                    verdict = "正确拒答" if r["top1_dist"] > th else "误召回"
        else:
            verdict = "命中" if (r["hit1"] or 0) > 0 else "未命中"
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            c["id"], c["cat"], r["gold_size"], fmt(r["hit1"], 2), fmt(r["mrr"]),
            fmt(r["recall"]), fmt(r["purity"]), fmt(r["top1_dist"], 4), verdict))

    print("\n".join(lines))

    print("\n---- 分类汇总 ----")
    summary = []
    for dt in ("kaoyan", "kaogong"):
        for cat in CATS:
            rs = [r for r in results if r["case"]["dt"] == dt and r["case"]["cat"] == cat]
            if not rs:
                continue
            if cat == "该空":
                ds = [r["top1_dist"] for r in rs if r["top1_dist"] is not None]
                row = (dt, cat, len(rs), "-", "-", "-", "-", dist_line(ds))
            else:
                row = (dt, cat, len(rs),
                       fmt(mean([r["hit1"] for r in rs]), 2),
                       fmt(mean([r["mrr"] for r in rs])),
                       fmt(mean([r["recall"] for r in rs])),
                       fmt(mean([r["purity"] for r in rs])), "-")
            summary.append(row)
    print("%-8s %-5s %-5s %-7s %-7s %-10s %-7s %s" %
          ("类型", "类别", "条数", "Hit@1", "MRR", "Recall", "纯度", "Top1距离分布"))
    for row in summary:
        print("%-8s %-5s %-5d %-7s %-7s %-10s %-7s %s" % row)

    print("\n注：字段类/大 gold 用例的 Recall@5 上限 = 5/gold（gold=240 时上限仅 2%），")
    print("    那是 K 的结构性上限、不是检索效果 —— 只读『纯度』和重建后的『过滤条数一致性』。")

    print("\n---- 拒答阈值标定（分类型）----")
    for dt in ("kaoyan", "kaogong"):
        hit_d = [r["top1_dist"] for r in results
                 if r["case"]["dt"] == dt and r["case"]["cat"] != "该空" and r["top1_dist"] is not None]
        empty_d = [r["top1_dist"] for r in results
                   if r["case"]["dt"] == dt and r["case"]["cat"] == "该空" and r["top1_dist"] is not None]
        print("%-8s 命中组 %s" % (dt, dist_line(hit_d)))
        print("%-8s 该空组 %s" % ("", dist_line(empty_d)))
        if hit_d and empty_d and max(hit_d) < min(empty_d):
            print("         → 可用阈值区间：(%.4f, %.4f)" % (max(hit_d), min(empty_d)))
        elif hit_d and empty_d:
            print("         → ⚠ 两组重叠：单一阈值无法分离（这就是你实测'正确答案比错误答案更远'的形态）")
            print("           → 实体类必须走 where_document 精确包含，不靠距离")

    fails = [r for r in results if r["err"] or (r["case"]["cat"] != "该空" and not (r["hit1"] or 0))]
    if fails:
        print("\n---- 未命中/错误明细（--show 看完整 Top-K）----")
        for r in fails:
            c = r["case"]
            print("\n[%s] %s  (%s, gold=%d)" % (c["id"], c["q"], c["cat"], r["gold_size"]))
            if c.get("note"):
                print("   备注：%s" % c["note"])
            if r["err"]:
                print("   错误：%s" % r["err"])
                continue
            for i, (key, dist, prev) in enumerate(r["retrieved"], 1):
                inG = "✓" if key in gold_keys(c) else " "
                print("   [%d]%s %.4f  %s | %s" % (i, inG, dist, key, prev))

    md = ["# RAG 评测结果 · %s" % args.tag, "",
          "- 时间：%s" % __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
          "- 模式：%s ｜ K=%d" % (args.mode, args.k),
          "- 距离空间：kaoyan=%s ｜ kaogong=%s（**L2 与 cosine 不可跨空间比较**）"
          % (space_of("kaoyan"), space_of("kaogong")),
          "- 数据：考研 %d 条 / 考公 %d 条 ｜ 评测集 %d 条" % (len(KY_ROWS), len(KG_ROWS), len(CASE)),
          "", "## 逐条", ""] + lines + ["", "## 分类汇总", "",
          "| 类型 | 类别 | 条数 | Hit@1 | MRR | Recall | 纯度 | Top1距离分布 |",
          "|---|---|---|---|---|---|---|---|"] + \
         ["| %s | %s | %d | %s | %s | %s | %s | %s |" % row for row in summary]

    out_dir = os.path.join(HERE, "eval", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_result_%s.md" % args.tag)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n已写出：%s" % out_path)
    print("把这张表贴进 EVAL.md（按 stage 分行），就是'用评测量化每轮改进'的证据。")


if __name__ == "__main__":
    main()
