"""P2 意图解析器 —— 按 RAG-P2-意图解析规则表 v4 实现"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS_PATH = os.path.join(HERE, "value_domains.json")

# ============ 常量 ============
NEGATION_WHITELIST = ["非全日制", "非全", "非法学", "非应届"]
NEGATION_MARKERS = ["不限", "不需要", "无需", "不要求", "除了", "没有"]
EDU_VERBS_USER = ["可报", "能报", "可考", "可以报", "可以考", "报考"]
EDU_VERBS_REQ = ["要求", "需要"]
# E: 排除形如年份的 4 位数（不当代码）
YEAR_LIKE = re.compile(r"(19|20)\d{2}")

# C: 兜底 contains 的停用词（含这些词的 n-gram 不当实体）
STOPWORDS = [
    "和", "或", "与", "及", "的", "了", "是",
    "哪些", "有哪些", "什么", "招什么", "吗", "呢", "哪",
    "岗位", "职位", "招聘", "工作",
    "可以", "要求", "需要", "报考", "能报", "可报",
    "事业单位", "机关",
    "四川省",                                     # 只能带"省"，不能写"四川"（会伤四川旅游学院）
    # ① 补：虚词/疑问词，"有没有招人" 这类反例靠它们拦
    "有", "没", "帮", "找", "看", "查", "给", "想", "要",
    "请问", "这个", "那个",
]

# ① 机构后缀闸门：pass 2 只放行"长得像机构名"的串（词表漏掉时兜底）
INSTITUTION_SUFFIXES = [
    "大学", "学院", "学校", "中学", "小学", "幼儿园",
    "中心", "研究院", "研究所",
    "医院", "卫生院",
    "局", "厅", "委", "部", "办", "署",
    "站", "馆", "园",
    "公司", "集团", "企业", "厂",
    "银行", "社",
    "队", "团", "会", "所",
]

# C: 已知维度关键词（含这些的 n-gram 不当实体）
DIM_KEYWORDS = ["本科", "研究生", "硕士", "博士", "大专", "中专",
                "党员", "共青团员", "政治面貌",
                "基层", "应届",
                "学硕", "专硕", "学术型", "专业型",
                "分数线", "复试线", "报录", "科目", "考什么",
                "非全", "全日制", "非全日制",
                "省属", "省直", "中央驻川"]

DIMENSION_ORDER = [
    "城市", "学历", "政治面貌", "基层", "年份",
    "学校", "学院", "学位类型", "字段类型", "学习方式",
    "实体", "专业", "代码", "unsupported",
]

_DOMAINS = None
def load_domains():
    global _DOMAINS
    if _DOMAINS is None:
        with open(DOMAINS_PATH, encoding="utf-8") as f:
            _DOMAINS = json.load(f)
    return _DOMAINS


# ============ 工具 ============
def _hanzi_count(s):
    return len(re.findall(r"[\u4e00-\u9fa5]", s))


def _mask_whitelist(query):
    consumed = [False] * len(query)
    for w in NEGATION_WHITELIST:
        idx = 0
        while True:
            pos = query.find(w, idx)
            if pos == -1:
                break
            for i in range(pos, pos + len(w)):
                consumed[i] = True
            idx = pos + 1
    return consumed


def find_negation_spans(query):
    consumed = _mask_whitelist(query)
    spans = []
    for marker in NEGATION_MARKERS:
        idx = 0
        while True:
            pos = query.find(marker, idx)
            if pos == -1:
                break
            if not any(consumed[pos:pos + len(marker)]):
                spans.append((marker, pos, pos + len(marker)))
            idx = pos + 1
    return spans


def is_negated(dim_start, neg_spans, query, window=2):
    for marker, ns, ne in neg_spans:
        if ne <= dim_start and _hanzi_count(query[ne:dim_start]) <= window:
            return True
    return False


def normalize_entity(s, dt, domains):
    """返回 (field, value) 或 None"""
    # G: 城市名归城市维度，不当机关/单位
    city_values = set(domains.get("city_alias", {}).values())
    if s in city_values:
        return None

    # ① 别名表
    alias = domains.get("alias_table", {}).get(dt, {})
    if s in alias:
        f, v = alias[s]
        return (f, v)

    # F: 招聘单位原值直接匹配（O(1)）
    known_units = set(domains.get("known_units", []))
    if s in known_units:
        return ("招聘单位", s)

    # ② 招录机关：前缀剥离 + 后缀补齐
    known = set(domains.get("known_agencies", []))
    for p in ("四川省", "四川"):
        if s.startswith(p):
            cand = s[len(p):]
            if cand in known:
                return ("招录机关", cand)
    hits = [a for a in known if a.endswith(s) and len(s) >= 3]
    if len(hits) == 1:
        return ("招录机关", hits[0])

    # 招聘单位变体表（去市/州后）
    uv = domains.get("units_variant", {})
    if s in uv:
        return ("招聘单位", uv[s])

    return None


def generate_ngrams(query, min_len=2, max_len=12):
    n = len(query)
    for length in range(max_len, min_len - 1, -1):
        for i in range(n - length + 1):
            yield query[i:i + length], i, i + length


# ============ 维度扫描器 ============
# 返回 list of (field, value, start, end, negated, op)
# op: "eq" | "$lte" | "$gte" | "$in" | "contains" | "unsupported"

def scan_city(query, dt, neg_spans, domains):
    if dt != "kaogong":
        return []
    suffix_set = set(INSTITUTION_SUFFIXES)
    results = []
    matched = []
    for short, full in sorted(domains.get("city_alias", {}).items(),
                              key=lambda x: -len(x[0])):
        idx = 0
        while True:
            pos = query.find(short, idx)
            if pos == -1:
                break
            if any(pos >= s and pos + len(short) <= e for s, e in matched):
                idx = pos + 1
                continue
            # 新增：城市词后面 6 字内出现机构后缀 → 是机构名的一部分，不当城市
            after = query[pos + len(short): pos + len(short) + 6]
            if any(suf in after for suf in suffix_set):
                idx = pos + 1
                continue
            neg = is_negated(pos, neg_spans, query)
            results.append(("城市", full, pos, pos + len(short), neg, "eq"))
            matched.append((pos, pos + len(short)))
            idx = pos + 1
    for kw, val in [("省属", "省属"), ("省直", "省属"), ("中央驻川", "中央驻川")]:
        pos = query.find(kw)
        if pos != -1:
            neg = is_negated(pos, neg_spans, query)
            results.append(("城市", val, pos, pos + len(kw), neg, "eq"))
    return results


def scan_education(query, dt, neg_spans, domains):
    if dt != "kaogong":
        return []
    results = []
    for kw, level in [("博士", 5), ("研究生", 4), ("硕士", 4),
                      ("本科", 3), ("大专", 2), ("中专", 1)]:
        idx = 0
        while True:
            pos = query.find(kw, idx)
            if pos == -1:
                break
            after = query[pos + len(kw):pos + len(kw) + 8]
            before = query[max(0, pos - 4):pos]
            is_user = any(v in after or v in before for v in EDU_VERBS_USER)
            is_req = any(v in after or v in before for v in EDU_VERBS_REQ)
            neg = is_negated(pos, neg_spans, query)

            if kw == "研究生" and (is_req or not is_user):
                results.append(("学历门槛", [4, 5], pos, pos + len(kw), neg, "$in"))
            elif kw in ("硕士", "博士") and (is_req or not is_user):
                results.append(("学历门槛", level, pos, pos + len(kw), neg, "eq"))
            elif kw == "本科":
                op = "$lte" if is_user else "eq"
                results.append(("学历门槛", 3, pos, pos + len(kw), neg, op))
            elif kw in ("大专", "中专"):
                op = "$lte" if is_user else "eq"
                results.append(("学历门槛", 2, pos, pos + len(kw), neg, op))
            idx = pos + 1
    return results


def scan_political(query, dt, neg_spans, domains):
    if dt != "kaogong":
        return []
    results = []
    # 先扫"政治面貌"
    pos = query.find("政治面貌")
    if pos != -1:
        neg = is_negated(pos, neg_spans, query)
        if neg:
            results.append(("政治面貌", "不限", pos, pos + 4, True, "eq"))
    # 党员
    for kw in ["中共党员", "党员"]:
        idx = 0
        while True:
            pos = query.find(kw, idx)
            if pos == -1:
                break
            neg = is_negated(pos, neg_spans, query)
            val = "不限" if neg else "中共党员"
            results.append(("政治面貌", val, pos, pos + len(kw), neg, "eq"))
            idx = pos + 1
    # 共青团员
    if "共青团员" in query:
        pos = query.find("共青团员")
        results.append(("__unsupported__", "共青团员", pos, pos + 4, False, "unsupported"))
    return results


def scan_grassroots(query, dt, neg_spans, domains):
    if dt != "kaogong":
        return []
    results = []
    for kw in ["2年以上基层", "两年以上基层", "两年基层"]:
        pos = query.find(kw)
        if pos != -1:
            results.append(("基层要求", "2年以上", pos, pos + len(kw), False, "eq"))
    for kw in ["有基层要求", "要求基层", "基层经历", "基层工作经历"]:
        pos = query.find(kw)
        if pos != -1:
            neg = is_negated(pos, neg_spans, query)
            if neg:
                results.append(("基层要求", "否", pos, pos + len(kw), True, "eq"))
            elif "2年" not in query and "两年" not in query:
                results.append(("基层要求", ["2年以上", "有要求(年限未明)"],
                               pos, pos + len(kw), False, "$in"))
    if "应届" in query:
        pos = query.find("应届")
        results.append(("基层要求", "否", pos, pos + 2, False, "eq"))
    return results


def scan_year(query, dt, neg_spans, domains):
    results = []
    for m in re.finditer(r"(20\d{2})", query):
        results.append(("年份", m.group(1), m.start(), m.end(), False, "eq"))
    for kw, yr in [("今年", "2026"), ("去年", "2025"), ("明年", "2027"),
                   ("最新", "2026"), ("最近", "2026")]:
        pos = query.find(kw)
        if pos != -1:
            results.append(("年份", yr, pos, pos + len(kw), False, "eq"))
    return results


def scan_school(query, dt, neg_spans, domains):
    if dt != "kaoyan":
        return []
    results = []
    for short, full in sorted(domains.get("school_alias", {}).items(),
                              key=lambda x: -len(x[0])):
        pos = query.find(short)
        if pos != -1:
            results.append(("学校", full, pos, pos + len(short), False, "eq"))
    return results


def scan_college(query, dt, neg_spans, domains):
    if dt != "kaoyan":
        return []
    results = []
    for c in domains.get("colleges", []):
        pos = query.find(c)
        if pos != -1:
            results.append(("学院", c, pos, pos + len(c), False, "eq"))
    return results


def scan_degree(query, dt, neg_spans, domains):
    if dt != "kaoyan":
        return []
    results = []
    for kw, val in [("学硕", "学硕"), ("学术型", "学硕"),
                    ("专硕", "专硕"), ("专业型", "专硕")]:
        pos = query.find(kw)
        if pos != -1:
            results.append(("学位类型", val, pos, pos + len(kw), False, "eq"))
    return results


def scan_field_type(query, dt, neg_spans, domains):
    if dt != "kaoyan":
        return []
    results = []
    for kw in ["分数线", "复试线", "多少分"]:
        pos = query.find(kw)
        if pos != -1:
            results.append(("字段类型", ["分数线", "分数线+招生目录"],
                           pos, pos + len(kw), False, "$in"))
            return results
    for kw in ["报录比", "报录", "报考人数"]:
        pos = query.find(kw)
        if pos != -1:
            results.append(("字段类型", "报录", pos, pos + len(kw), False, "eq"))
            return results
    for kw in ["考什么", "考试科目", "科目"]:
        pos = query.find(kw)
        if pos != -1:
            results.append(("字段类型", ["招生目录", "分数线+招生目录"],
                           pos, pos + len(kw), False, "$in"))
            return results
    return results


def scan_study_mode(query, dt, neg_spans, domains):
    results = []
    if dt == "kaogong":
        # 考公侧无此维度：命中任一学习方式词 → unsupported
        for kw in ["非全日制", "非全", "全日制"]:
            pos = query.find(kw)
            if pos != -1:
                results.append(("__unsupported__", "学习方式",
                                pos, pos + len(kw), False, "unsupported"))
                return results
        return results

    # 考研侧
    for kw in ["非全日制", "非全"]:
        pos = query.find(kw)
        if pos != -1:
            results.append(("学习方式", "非全日制", pos, pos + len(kw), False, "eq"))
            return results
    m = re.search(r"(?<!非)全日制", query)
    if m:
        results.append(("__unsupported__", "学习方式=全日制",
                        m.start(), m.end(), False, "unsupported"))
    return results


def scan_entity_exact(query, dt, neg_spans, domains, consumed):
    """pass 1：①② 命中（别名 / 机关 / 单位变体）—— 精确，走 where"""
    results = []
    for ngram, start, end in generate_ngrams(query, 2, 12):
        if any(start < e and end > s for s, e in consumed):
            continue
        hit = normalize_entity(ngram, dt, domains)
        if hit:
            f, v = hit
            results.append((f, v, start, end, False, "eq"))
            consumed.append((start, end))
    return results


def scan_entity_fallback(query, dt, neg_spans, domains, consumed):
    """pass 2：③ 兜底 contains —— **所有维度扫完后**再跑，顺序依赖消失

    双闸门（必须同时满足）：
      - 区间排除：不与 DIM_KEYWORDS / STOPWORDS / 已消费 重叠
      - 机构后缀：必须以 INSTITUTION_SUFFIXES 之一结尾
    """
    exclude = []
    for kw in DIM_KEYWORDS + STOPWORDS:
        idx = 0
        while True:
            pos = query.find(kw, idx)
            if pos == -1:
                break
            exclude.append((pos, pos + len(kw)))
            idx = pos + 1

    major_kws = set(domains.get("major_keywords", []))
    results = []
    for ngram, start, end in generate_ngrams(query, 4, 12):
        if any(start < e and end > s for s, e in consumed):
            continue
        if any(start < e and end > s for s, e in exclude):
            continue
        if not re.fullmatch(r"[\u4e00-\u9fa5]+", ngram):
            continue
        if not any(ngram.endswith(suf) for suf in INSTITUTION_SUFFIXES):
            continue
        if ngram in major_kws or any(kw in ngram for kw in major_kws):
            continue
        results.append(("__contains__", ngram, start, end, False, "contains"))
        consumed.append((start, end))
    return results


def scan_major(query, dt, neg_spans, domains, consumed):
    """专业维度：**先精确（长）后泛词（短）**

    ① 考研：用 kaoyan_majors（库内专业值域，约 40 个）反查 → {"专业": X}，走 where 精确
    ② 泛词兜底：major_keywords（计算机/会计…）→ contains

    ⚠ 顺序不能反：若"计算机"这类泛词先消费，会吃掉"计算机科学与技术"的头部，
      导致精确专业名永远匹配不上。
    """
    results = []

    # ① 考研专业值域反查（长优先，走 where 精确）
    if dt == "kaoyan":
        for m in sorted(domains.get("kaoyan_majors", []), key=lambda x: -len(x)):
            if len(m) < 2:
                continue
            idx = 0
            while True:
                pos = query.find(m, idx)
                if pos == -1:
                    break
                if any(pos < e and pos + len(m) > s for s, e in consumed):
                    idx = pos + 1
                    continue
                results.append(("专业", m, pos, pos + len(m), False, "eq"))
                consumed.append((pos, pos + len(m)))
                idx = pos + 1

    # ② 泛词兜底（短，走 contains）—— "计算机" 这类非完整专业名
    for kw in domains.get("major_keywords", []):
        idx = 0
        while True:
            pos = query.find(kw, idx)
            if pos == -1:
                break
            if any(pos < e and pos + len(kw) > s for s, e in consumed):
                idx = pos + 1
                continue
            results.append(("__contains__", kw, pos, pos + len(kw), False, "contains"))
            consumed.append((pos, pos + len(kw)))
            idx = pos + 1
    return results


def scan_code(query, dt, neg_spans, domains, consumed):
    results = []
    if dt == "kaogong":
        for m in re.finditer(r"\d{6,}", query):
            results.append(("职位代码", m.group(), m.start(), m.end(), False, "eq"))
    else:  # kaoyan
        # ④ 不依赖 \w 语义（Python re 把 CJK 当 \w → \b 在中文里不生效）
        for m in re.finditer(r"(?<!\d)\d{4,6}(?!\d)", query):
            d = m.group()
            if YEAR_LIKE.fullmatch(d):
                continue
            results.append(("代码", d, m.start(), m.end(), False, "eq"))
    return results

def scan_unsupported(query, dt, neg_spans, domains):
    """识别出但字段不支持 → unsupported（不静默丢）"""
    results = []

    # 招 X 人以上
    m = re.search(r"招\s*(\d+)\s*人", query)
    if m:
        n = m.group(1)
        results.append(("__unsupported__", f"招录人数≥{n}",
                        m.start(), m.end(), False, "unsupported"))

    # 年龄
    m = re.search(r"(\d+\s*岁以下|年龄要求|年龄限制)", query)
    if m:
        results.append(("__unsupported__", "年龄要求",
                        m.start(), m.end(), False, "unsupported"))

    # 进面比例
    m = re.search(r"(进面比例|竞争小|面试比例)", query)
    if m:
        results.append(("__unsupported__", "进面比例",
                        m.start(), m.end(), False, "unsupported"))

    # 工作地点
    m = re.search(r"(工作地点|上班地点|工作地址)", query)
    if m:
        results.append(("__unsupported__", "工作地点",
                        m.start(), m.end(), False, "unsupported"))

    # 专业不限 / 学历不限
    if "专业不限" in query or "不限专业" in query:
        pos = query.find("专业不限") if "专业不限" in query else query.find("不限专业")
        results.append(("__unsupported__", "专业不限", pos, pos + 4, False, "unsupported"))
    if "学历不限" in query or "不限学历" in query:
        pos = query.find("学历不限") if "学历不限" in query else query.find("不限学历")
        results.append(("__unsupported__", "学历不限", pos, pos + 4, False, "unsupported"))

    return results


# ============ 派生抑制 ============
def apply_suppression(collected):
    fields = {c[0] for c in collected}
    if "代码" in fields:
        collected[:] = [c for c in collected if c[0] not in ("专业", "学位类型")]


# ============ 输出规范化 ============
def normalize_output(collected):
    """塌缩规则 + 递归规范化"""
    where = {}
    contains = []
    unsupported = []
    and_clauses = []

    for field, value, start, end, negated, op in collected:
        if field == "__unsupported__":
            unsupported.append(value)
            continue
        if field == "__contains__":
            contains.append(value)
            continue

        # 否定城市：$ne + $nin
        if negated and field == "城市":
            and_clauses.append({"城市": {"$ne": value}})
            and_clauses.append({"城市": {"$nin": ["省属", "中央驻川", "国企"]}})
            continue

        # B: 同字段多值 → 通用升级为 $in（支持 eq→eq、eq→$in、$in→eq）
        if field in where:
            prev = where[field]
            if isinstance(prev, dict) and "$in" in prev:
                vals = list(prev["$in"])
            elif isinstance(prev, (str, int, float)):
                vals = [prev]
            else:
                # 已有比较操作符（$lte/$gte/$ne/$nin）→ 不合并，保留
                continue
            if value not in vals:
                vals.append(value)
            where[field] = {"$in": vals} if len(vals) > 1 else vals[0]
            continue

        if op == "eq":
            where[field] = value
        elif op in ("$lte", "$gte"):
            where[field] = {op: value}
        elif op == "$in":
            where[field] = {"$in": list(value) if isinstance(value, (list, tuple)) else [value]}

    # 合并 where + and_clauses
    if and_clauses:
        for k, v in where.items():
            and_clauses.append({k: v})
        where = {"$and": and_clauses}

    if "$and" not in where and len(where) > 1:
        where = {"$and": [{k: v} for k, v in where.items()]}
    # 塌缩：单项 $and
    if "$and" in where and len(where["$and"]) == 1:
        where = where["$and"][0]
    # 塌缩：单值 $in
    for k, v in list(where.items()):
        if isinstance(v, dict) and "$in" in v and len(v["$in"]) == 1:
            where[k] = v["$in"][0]
    contains_op = "or"
    return where or None, contains or None, unsupported, contains_op


def decide_strategy(where, contains, unsupported):
    if unsupported and not (where or contains):
        return "reject"
    if where or contains:
        return "filtered"
    return "semantic"


# ============ 主入口 ============
def parse_intent(query, data_type):
    domains = load_domains()
    neg_spans = find_negation_spans(query)
    collected = []
    consumed = []

    scanners = {
        "城市": lambda: scan_city(query, data_type, neg_spans, domains),
        "学历": lambda: scan_education(query, data_type, neg_spans, domains),
        "政治面貌": lambda: scan_political(query, data_type, neg_spans, domains),
        "基层": lambda: scan_grassroots(query, data_type, neg_spans, domains),
        "年份": lambda: scan_year(query, data_type, neg_spans, domains),
        "学校": lambda: scan_school(query, data_type, neg_spans, domains),
        "学院": lambda: scan_college(query, data_type, neg_spans, domains),
        "学位类型": lambda: scan_degree(query, data_type, neg_spans, domains),
        "字段类型": lambda: scan_field_type(query, data_type, neg_spans, domains),
        "学习方式": lambda: scan_study_mode(query, data_type, neg_spans, domains),
        "实体": lambda: scan_entity_exact(query, data_type, neg_spans, domains, consumed),
        "专业": lambda: scan_major(query, data_type, neg_spans, domains, consumed),
        "代码": lambda: scan_code(query, data_type, neg_spans, domains, consumed),
        "unsupported": lambda: scan_unsupported(query, data_type, neg_spans, domains),
    }

    for dim in DIMENSION_ORDER:
        collected.extend(scanners[dim]())

    # ③ 收尾 pass：所有维度扫完后再做兜底 contains
    collected.extend(scan_entity_fallback(query, data_type, neg_spans, domains, consumed))

    apply_suppression(collected)
    where, contains, unsupported, contains_op = normalize_output(collected)

    return {
        "data_type": data_type,
        "where": where,
        "contains": contains,
        "contains_op": contains_op,               # ⑥ 执行端不再猜
        "strategy": decide_strategy(where, contains, unsupported),
        "reason": "",
        "unsupported": unsupported,
        "warning": [],
        "sort": None,
        "limit": None,
    }

def build_where_document(contains, op="or"):
    """把 contains 列表组装成 where_document 结构（§12.6）

    - 空 → None
    - 单条 → {"$contains": ...}
    - 多条 → {"$or"/"$and": [{"$contains": ...}, ...]}
    """
    if not contains:
        return None
    if len(contains) == 1:
        return {"$contains": contains[0]}
    return {f"${op}": [{"$contains": c} for c in contains]}