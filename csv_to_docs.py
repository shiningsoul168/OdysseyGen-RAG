import csv
import os
import re
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

DB_DIR = "./chroma_db"
COLLECTIONS = {"kaoyan": "kaoyan", "kaogong": "kaogong"}

DATA_SOURCES = [
    {"path": "kaoyan_data.csv", "data_type": "kaoyan"},
    {"path": "kaogong_data.csv", "data_type": "kaogong"},
]

# ---- 全局收集：静默错误的入口 ----
UNMAPPED_EDU = []          # 学历未识别
UNMAPPED_CITY = []         # 城市未覆盖
DIRTY_ROWS = []            # 推免 > 招生


# ============================================================
# 考研
# ============================================================
def classify_kaoyan(row):
    has_score = bool((row.get("总分") or "").strip())
    has_subject = bool((row.get("考试科目") or "").strip())
    has_apply = bool((row.get("统考报考") or "").strip())
    if has_score and has_subject:
        return "分数线+招生目录"
    if has_score:
        return "分数线"
    if has_subject:
        return "招生目录"
    if has_apply:
        return "报录"
    return None


def parse_kaoyan_remark(remark, row_type):
    """按 ； 切分，分流到复试科目/学习方式/分数线说明/报录说明/备注"""
    out = {"复试科目": "", "学习方式": "", "分数线说明": "", "报录说明": "", "备注": ""}
    if not remark:
        return out
    for seg in (s.strip() for s in remark.split("；")):
        if not seg:
            continue
        if seg == "2026复试线":
            continue
        if "复试科目" in seg:
            out["复试科目"] = seg.split("：", 1)[-1].strip()
        elif seg.startswith("复试："):
            out["复试科目"] = seg.split("：", 1)[1].strip()
        elif "非全" in seg:
            out["学习方式"] = seg
        elif row_type == "报录":
            out["报录说明"] = seg
        elif "单科按校线" in seg or "学院总分上调" in seg:
            out["分数线说明"] = seg
        else:
            out["备注"] = seg
    return out


def build_kaoyan(row):
    school = (row.get("学校") or "").strip()
    college = (row.get("学院") or "").strip()
    major = (row.get("专业") or "").strip()
    code = (row.get("代码") or "").strip()
    degree = (row.get("学位类型") or "").strip()
    year = (row.get("年份") or "").strip()
    remark = (row.get("备注") or "").strip()
    source = (row.get("数据来源") or "").strip()

    row_type = classify_kaoyan(row)
    if row_type is None:
        return None

    has_score = bool((row.get("总分") or "").strip())
    has_subject = bool((row.get("考试科目") or "").strip())
    has_apply = bool((row.get("统考报考") or "").strip())

    tags = []
    if has_score:
        tags.append("复试分数线")
    if has_subject:
        tags.append("招生目录")
    if has_apply:
        tags.append("报录数据")

    lines = [f"【本条含】{'、'.join(tags)}"]

    id_line = f"【学校】{school}"
    if college:
        id_line += f"【学院】{college}"
    id_line += f"【专业】{major}【代码】{code}"
    lines.append(id_line)
    lines.append(f"【学位类型】{degree}【年份】{year}")

    if has_score:
        scores = []
        for key in ("总分", "政治", "外语", "数学", "专业课"):
            val = (row.get(key) or "").strip()
            if val:
                scores.append(f"{key}{val}")
        lines.append(f"【复试分数线】{'、'.join(scores)}")

    if has_subject:
        lines.append(f"【考试科目】{(row.get('考试科目') or '').strip()}")

    enroll_str = (row.get("招生人数") or "").strip()
    recommend_str = (row.get("推免人数") or "").strip()
    if enroll_str:
        e = int(enroll_str)
        if recommend_str:
            r = int(recommend_str)
            if r > e:
                DIRTY_ROWS.append((code, e, r))
            lines.append(f"【招生人数】{e}人（含推免{r}人，统考名额约{e - r}人）")
        else:
            lines.append(f"【招生人数】{e}人")

    tuition = (row.get("学费") or "").strip()
    if tuition:
        lines.append(f"【学费】{tuition}元/年")
    duration = (row.get("学制") or "").strip()
    if duration:
        lines.append(f"【学制】{duration}年")

    parsed = parse_kaoyan_remark(remark, row_type)
    if parsed["复试科目"]:
        lines.append(f"【复试科目】{parsed['复试科目']}")
    if parsed["学习方式"]:
        lines.append(f"【学习方式】{parsed['学习方式']}")
    if parsed["分数线说明"]:
        lines.append(f"【分数线说明】{parsed['分数线说明']}")

    if has_apply:
        parts_r = []
        for k in ("统考报考", "推免录取", "统考录取"):
            v = (row.get(k) or "").strip()
            if v:
                parts_r.append(f"{k}{v}人")
        lines.append(f"【报录数据】{'、'.join(parts_r)}")

    if parsed["报录说明"]:
        lines.append(f"【报录说明】{parsed['报录说明']}")
    if parsed["备注"]:
        lines.append(f"【备注】{parsed['备注']}")

    content = "\n".join(lines)

    metadata = {
        "data_type": "kaoyan",
        "学校": school,
        "专业": major,
        "代码": code,
        "学位类型": degree,
        "年份": year,
        "字段类型": row_type,
        "来源": source,
    }
    if college:
        metadata["学院"] = college
    if parsed["学习方式"]:
        metadata["学习方式"] = parsed["学习方式"]

    return Document(page_content=content, metadata=metadata)


# ============================================================
# 考公
# ============================================================
CITY_OVERRIDE = {
    "国家粮食和物资储备局四川局": "中央驻川",
    "成都海关": "中央驻川",
    "四川省通信管理局": "中央驻川",
    "四川电信实业集团有限责任公司": "国企",
    "团省委": "省属",
    "四川广播电视台": "省属",
    "审计厅": "省属",
}


def derive_city(agency):
    if agency in CITY_OVERRIDE:
        return CITY_OVERRIDE[agency]
    if agency.endswith(("市", "州")):
        return agency
    if agency.startswith("省"):
        return "省属"
    raise ValueError(f"未覆盖机关: {agency}")


def edu_level(v):
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


def political_norm(v):
    v = (v or "").strip()
    if not v:
        return "不限"
    if "中共" in v:
        return "中共党员"
    return v


def grassroots_norm(v):
    v = (v or "").strip()
    if not v:
        return "否"
    if v == "2年以上基层工作经历":
        return "2年以上"
    return "有要求(年限未明)"


def major_clean(raw):
    """换行 → 中文分号，去掉多余空白"""
    return "；".join(s.strip() for s in raw.splitlines() if s.strip())


def extract_major_kw(text):
    kws = list(set(re.findall(r"[\u4e00-\u9fa5]{2,}类", text)))
    return "、".join(kws)


def build_kaogong(row):
    province = (row.get("省份") or "").strip()
    year = (row.get("年份") or "").strip()
    exam_type = (row.get("考试类型") or "").strip()
    agency = (row.get("招录机关") or "").strip()
    unit = (row.get("招聘单位") or "").strip()
    position = (row.get("职位名称") or "").strip()
    code = (row.get("职位代码") or "").strip()
    category = (row.get("岗位类别") or "").strip()
    headcount = (row.get("招录人数") or "").strip()
    edu_raw = (row.get("学历要求") or "").strip()
    major_raw = (row.get("专业要求") or "").strip()
    political_raw = (row.get("政治面貌") or "").strip()
    grassroots_raw = (row.get("基层年限") or "").strip()
    age = (row.get("年龄要求") or "").strip()
    interview = (row.get("面试比例") or "").strip()
    remark = (row.get("备注") or "").strip()
    source = (row.get("数据来源") or "").strip()

    city = derive_city(agency)
    major = major_clean(major_raw)

    lines = []
    lines.append(f"【省份】{province}【年份】{year}【考试类型】{exam_type}")
    id_line = (f"【招录机关】{agency}【招聘单位】{unit}"
               f"【职位名称】{position}【职位代码】{code}")
    if category:
        id_line += f"【岗位类别】{category}"
    if city:
        id_line += f"【城市】{city}"
    lines.append(id_line)

    if headcount:
        lines.append(f"【招录人数】{headcount}")
    if edu_raw:
        lines.append(f"【学历要求】{edu_raw}")
    if major:
        lines.append(f"【专业要求】{major}")

    if political_raw:
        lines.append(f"【政治面貌】{political_raw}")
    else:
        lines.append("【政治面貌】不限（公告未列明）")

    if grassroots_raw:
        lines.append(f"【基层年限】{grassroots_raw}")
    else:
        lines.append("【基层年限】无要求")

    if age:
        lines.append(f"【年龄要求】{age}")
    if interview:
        lines.append(f"【面试比例】{interview}")
    if remark:
        lines.append(f"【其他要求】{remark}")

    content = "\n".join(lines)

    lv = edu_level(edu_raw)
    metadata = {
        "data_type": "kaogong",
        "省份": province,
        "年份": year,
        "考试类型": exam_type,
        "城市": city,
        "招录机关": agency,
        "招聘单位": unit,
        "职位名称": position,
        "职位代码": code,
        "政治面貌": political_norm(political_raw),
        "基层要求": grassroots_norm(grassroots_raw),
        "来源": source,
    }
    if category:
        metadata["岗位类别"] = category
    if lv > 0:
        metadata["学历门槛"] = lv
    else:
        UNMAPPED_EDU.append((code, edu_raw))

    kw = extract_major_kw(major)
    if kw:
        metadata["专业关键词"] = kw

    if remark:
        metadata["其他要求"] = remark

    return Document(page_content=content, metadata=metadata)


BUILDERS = {"kaoyan": build_kaoyan, "kaogong": build_kaogong}


def load_csv_docs(path, data_type):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    docs = []
    skipped = 0
    for r in rows:
        d = BUILDERS[data_type](r)
        if d is None:
            skipped += 1
            continue
        docs.append(d)
    print(f"[load] {path} -> {len(docs)} 条 ({data_type}), 跳过 {skipped}")
    return docs


# ============================================================
# 建库
# ============================================================
def reset_db():
    import chromadb
    if not os.path.exists(DB_DIR):
        return
    client = chromadb.PersistentClient(path=DB_DIR)
    existing = [c.name for c in client.list_collections()]
    print(f"[2] 待删除 collection: {existing}")
    for n in existing:
        client.delete_collection(n)
    remaining = [c.name for c in client.list_collections()]
    print(f"[2] 删除后剩余: {remaining}")
    assert remaining == [], "旧 collection 未清空"
    del client
    import gc
    gc.collect()


def main():
    all_docs = {"kaoyan": [], "kaogong": []}
    for src in DATA_SOURCES:
        if not os.path.exists(src["path"]):
            print(f"[skip] 找不到 {src['path']}")
            continue
        all_docs[src["data_type"]].extend(
            load_csv_docs(src["path"], src["data_type"])
        )

    # ---- 断言 1：考研四分类 ----
    from collections import Counter
    ky_types = Counter(d.metadata["字段类型"] for d in all_docs["kaoyan"])
    print(f"\n[断言1] 考研分类: {dict(ky_types)}  总计={sum(ky_types.values())}")
    assert sum(ky_types.values()) == 69, "考研条数不对"

    # ---- 断言 2：考公城市 ----
    kg_cities = Counter(d.metadata["城市"] for d in all_docs["kaogong"])
    city_groups = {"市州": 0, "省属": 0, "中央驻川": 0, "国企": 0}
    for c, n in kg_cities.items():
        if c in ("省属", "中央驻川", "国企"):
            city_groups[c] = n
        else:
            city_groups["市州"] += n
    print(f"[断言2] 考公城市: {city_groups}  总计={sum(city_groups.values())}")
    assert city_groups["市州"] == 5027, f"市州 {city_groups['市州']} != 5027"
    assert city_groups["省属"] == 530, f"省属 {city_groups['省属']} != 530"
    assert city_groups["中央驻川"] == 78, f"中央驻川 {city_groups['中央驻川']} != 78"
    assert city_groups["国企"] == 14, f"国企 {city_groups['国企']} != 14"

    # ---- 打印静默问题 ----
    if UNMAPPED_EDU:
        print(f"\n[学历未识别 {len(UNMAPPED_EDU)} 条]（人工看）")
        for code, raw in UNMAPPED_EDU:
            print(f"  {code}: {raw!r}")
    if DIRTY_ROWS:
        print(f"\n[脏行 推免>招生 {len(DIRTY_ROWS)} 条]")
        for code, e, r in DIRTY_ROWS:
            print(f"  {code}: 招生{e} 推免{r}")

    reset_db()

    embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    for dt, name in COLLECTIONS.items():
        docs = all_docs[dt]
        if not docs:
            print(f"[skip] {dt} 无数据")
            continue
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=name,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=DB_DIR,
        )
        print(f"[3] {name}: 已写入 {len(docs)} 条 (cosine)")


if __name__ == "__main__":
    main()