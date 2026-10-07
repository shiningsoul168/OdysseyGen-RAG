"""从 Chroma 库导出值域到 value_domains.json（§12.8）"""
import json
import chromadb

DB_DIR = "./chroma_db"

def export():
    client = chromadb.PersistentClient(path=DB_DIR)
    domains = {}

    # 考公
    kg = client.get_collection("kaogong")
    kg_meta = kg.get(include=["metadatas"])["metadatas"]

    domains["known_agencies"] = sorted({m.get("招录机关") for m in kg_meta if m.get("招录机关")})
    units = sorted({m.get("招聘单位") for m in kg_meta if m.get("招聘单位")})
    domains["known_units"] = units  # 4017 个招聘单位原值，用于 normalize_entity 直接匹配
    domains["units_variant"] = {}
    for u in units:
        if "市" in u or "州" in u:
            v = u.replace("市", "").replace("州", "")
            if v != u:
                if v in domains["units_variant"]:
                    print(f"[冲突] {v}: {domains['units_variant'][v]} vs {u} → 不生成")
                else:
                    domains["units_variant"][v] = u

    domains["city_alias"] = {
        "成都": "成都市", "凉山": "凉山州", "阿坝": "阿坝州", "甘孜": "甘孜州",
        "广安": "广安市", "广元": "广元市", "绵阳": "绵阳市", "南充": "南充市",
        "自贡": "自贡市", "内江": "内江市", "德阳": "德阳市", "达州": "达州市",
        "乐山": "乐山市", "眉山": "眉山市", "宜宾": "宜宾市", "遂宁": "遂宁市",
        "泸州": "泸州市", "雅安": "雅安市", "资阳": "资阳市",
        "攀枝花": "攀枝花市", "巴中": "巴中市",
    }

    domains["major_keywords"] = ["计算机", "会计", "新闻传播", "法学", "金融"]

    domains["alias_table"] = {
        "kaogong": {"广安融媒": ["招聘单位", "广安市融媒体中心"]},
        "kaoyan": {
            "川大":     ["学校", "四川大学"],
            "电子科大": ["学校", "电子科技大学"],
            "成电":     ["学校", "电子科技大学"],
            "西南交大": ["学校", "西南交通大学"],
            "成都理工": ["学校", "成都理工大学"],
            "计科":     ["代码", "081200"],          # ← 从专业改到代码
            "软工":     ["专业", "软件工程"],
            "软工专硕": ["代码", "085405"],
            "软工学硕": ["代码", "083500"],
        },
    }

    # 考研
    ky = client.get_collection("kaoyan")
    ky_meta = ky.get(include=["metadatas"])["metadatas"]
    domains["colleges"] = sorted({m.get("学院") for m in ky_meta if m.get("学院")})
    # 专业值域：给 scan_major 反查用 —— 比手工维护 major_keywords 可靠
    # （起因：真实 query「四川大学 软件工程 考研 分数线 考试科目」里"软件工程"被丢弃，
    #   因为 major_keywords 只有 5 个手写词。29 条解析回归 + 40 条评测集都没测真实 query）
    domains["kaoyan_majors"] = sorted({m.get("专业") for m in ky_meta if m.get("专业")})
    domains["school_alias"] = {"川大": "四川大学"}   # 当前只有一所

    with open("value_domains.json", "w", encoding="utf-8") as f:
        json.dump(domains, f, ensure_ascii=False, indent=2)

    print(f"导出完成：")
    print(f"  known_agencies: {len(domains['known_agencies'])}")
    print(f"  units_variant: {len(domains['units_variant'])}")
    print(f"  city_alias: {len(domains['city_alias'])}")
    print(f"  colleges: {domains['colleges']}")
    print(f"  kaoyan_majors: {len(domains['kaoyan_majors'])}   # 应约 40（考研专业值域，给 scan_major 反查）")
    print(f"  known_units: {len(domains['known_units'])}")


if __name__ == "__main__":
    export()