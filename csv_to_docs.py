import csv
import os
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

CSV_PATH = "kaoyan_data_new.csv"
DB_DIR = "./chroma_db"


def row_to_document(row: dict) -> Document:
    """把 CSV 的一行转成一个 Document（一行 = 一个检索单元）"""
    school = (row.get("学校") or "").strip()
    college = (row.get("学院") or "").strip()
    major = (row.get("专业") or "").strip()
    code = (row.get("代码") or "").strip()
    degree = (row.get("学位类型") or "").strip()
    year = (row.get("年份") or "").strip()
    remark = (row.get("备注") or "").strip()

    # ---- 标题：学校 + 学院 + 年份 + 学位类型 + 专业 + 代码 ----
    head = f"{school}{college}"
    if year:
        head += f"{year}年"
    head += f"{degree}{major}"
    if code:
        head += f"（专业代码{code}）"

    parts = [head]

    # ---- 复试分数线（跳过空值） ----
    score_fields = [
        ("总分", "总分"),
        ("政治", "政治"),
        ("外语", "外语"),
        ("数学", "数学"),
        ("专业课", "专业课"),
    ]
    scores = []
    for key, label in score_fields:
        val = (row.get(key) or "").strip()
        if val:
            scores.append(f"{label}{val}分")
    if scores:
        parts.append("复试分数线" + "、".join(scores))

    # ---- 考试科目 ----
    exam_subjects = (row.get("考试科目") or "").strip()
    if exam_subjects:
        parts.append(f"考试科目{exam_subjects}")

    # ---- 招生人数 + 推免 ----
    enroll = (row.get("招生人数") or "").strip()
    recommend = (row.get("推免人数") or "").strip()
    if enroll:
        info = f"招生{enroll}人"
        if recommend:
            info += f"（含推免{recommend}人）"
        parts.append(info)

    # ---- 学费 + 学制 ----
    tuition = (row.get("学费") or "").strip()
    duration = (row.get("学制") or "").strip()
    if tuition or duration:
        info = ""
        if tuition:
            info += f"学费{tuition}元/年"
        if duration:
            info += f"，学制{duration}年"
        parts.append(info.strip("，"))

    # ---- 报录数据（跳过空值，只有填了才拼） ----
    apply_num = (row.get("统考报考") or "").strip()
    recommend_num = (row.get("推免录取") or "").strip()
    admit_num = (row.get("统考录取") or "").strip()
    if apply_num or recommend_num or admit_num:
        parts.append(
            f"统考报考{apply_num or '—'}人，"
            f"推免录取{recommend_num or '—'}人，"
            f"统考录取{admit_num or '—'}人"
        )

    # ---- 备注 ----
    if remark:
        parts.append(f"备注{remark}")

    # ---- 最终拼接：标题独立成句，后面用逗号连成一句 ----
    if len(parts) == 1:
        content = parts[0] + "。"
    else:
        content = parts[0] + "。" + "，".join(parts[1:]) + "。"

    metadata = {
        "学校": school,
        "学院": college,
        "专业": major,
        "代码": code,
        "学位类型": degree,
        "年份": year,
        "数据来源": (row.get("数据来源") or "").strip(),
    }
    return Document(page_content=content, metadata=metadata)


def main():
    # ---- 1. 读 CSV ----
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    docs = [row_to_document(r) for r in rows]
    print(f"[1] 共读取 {len(docs)} 条数据")

    print("\n--- 示例 page_content（第1条）---")
    print(docs[0].page_content)
    print("--- 示例 metadata ---")
    print(docs[0].metadata)

    print("\n--- 示例 page_content（管理类，无政治/无专业课二）---")
    # 工商管理在第32条附近，这里直接找
    for d in docs:
        if "工商管理" in d.metadata["专业"]:
            print(d.page_content)
            break

    # ---- 2. 删除旧库，防止新旧数据混在一起 ----
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"\n[2] 已删除旧库 {DB_DIR}")

    # ---- 3. 向量化并写入 Chroma ----
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"[3] 已写入 {DB_DIR}，共 {vectordb._collection.count()} 条向量")

    # ---- 4. 检索验证 ----
    queries = [
        "川大计算机学院软件工程考什么科目？",
        "川大计算机科学与技术招生多少人？",
        "川大电子信息分数线多少？",
    ]
    for q in queries:
        print(f"\n=== 查询：{q} ===")
        results = vectordb.similarity_search(q, k=3)
        for i, d in enumerate(results, 1):
            print(f"[{i}] {d.page_content}")
            print(f"    metadata: {d.metadata}")


if __name__ == "__main__":
    main()