from rag_core import vectordb

# 分别测：明显该命中的、明显不该命中的
cases = [
    ("命中", "川大计算机科学与技术分数线多少？", "kaoyan"),
    ("命中", "广安融媒体中心招全媒体记者吗？", "kaogong"),
    ("命中", "招聘要求党员身份的岗位？", "kaogong"),
    ("该空", "四川旅游学院的岗位有哪些？", "kaogong"),
    ("命中", "川大医学院临床医学分数线？", "kaoyan"),
    ("该空", "深圳有哪些公务员岗位？", "kaogong"),
]

for label, q, dt in cases:
    print(f"\n=== [{label}] {q} ===")
    results = vectordb.similarity_search_with_score(q, k=5, filter={"data_type": dt})
    for i, (doc, score) in enumerate(results, 1):
        head = doc.page_content[:50]
        print(f"[{i}] score={score:.4f}  {head}...")