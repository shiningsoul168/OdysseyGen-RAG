# build_gongkao_csv.py
import re
from pathlib import Path

import pandas as pd

BASE = Path(r"F:\OdysseyGen-RAG")

FILE_CITY = BASE / "data" / "raw" / "2026年上半年市（州）事业单位公开招聘工作人员岗位和条件要求一览表.xlsx"
FILE_PROV = BASE / "data" / "raw" / "2026年上半年省属事业单位公开招聘工作人员岗位和条件要求一览表.xlsx"
FILE_TEACHER = BASE / "data" / "raw" / "2026年上半年全省公开招聘中小学教师（省属）岗位和条件要求一览表.xlsx"

CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
          "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def clean_id(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def extract_political(text):
    if pd.isna(text):
        return ""
    text = str(text)
    if "中共党员" in text:
        if "预备" in text:
            return "中共党员（含预备党员）"
        return "中共党员"
    if "共青团员" in text:
        return "共青团员"
    return ""


def extract_grassroots(text):
    """兼容阿拉伯数字和汉字数字，兼容'年以上'和'年及以上'"""
    if pd.isna(text):
        return ""
    text = str(text)

    # 阿拉伯数字：2年以上 / 2年及以上
    m = re.search(r"(\d+)\s*年及?以上基层工作经历", text)
    if m:
        return f"{m.group(1)}年以上基层工作经历"

    # 汉字数字：两年以上 / 两年及以上
    m = re.search(r"([一两二三四五六七八九十])\s*年及?以上基层工作经历", text)
    if m:
        return f"{CN_NUM[m.group(1)]}年以上基层工作经历"

    # 兜底：提到基层工作经历但没写年限
    if "基层工作经历" in text:
        return "有基层工作经历要求"

    return ""


def to_standard(df, exam_type, source_file, agency_col):
    rows = []
    for _, r in df.iterrows():
        other = r.get("其他", "")
        if pd.isna(other):
            other = ""

        rows.append({
            "data_type": "kaogong",
            "省份": "四川",
            "年份": "2026",
            "考试类型": exam_type,
            "招录机关": str(r.get(agency_col, "")).strip() if pd.notna(r.get(agency_col)) else "",
            "招聘单位": str(r.get("招聘单位", "")).strip() if pd.notna(r.get("招聘单位")) else "",
            "职位名称": str(r.get("岗位名称", "")).strip() if pd.notna(r.get("岗位名称")) else "",
            "职位代码": clean_id(r.get("岗位编码", "")),
            "岗位类别": str(r.get("岗位类别", "")).strip() if pd.notna(r.get("岗位类别")) else "",
            "招录人数": str(r.get("招聘人数", "")).strip() if pd.notna(r.get("招聘人数")) else "",
            "学历要求": str(r.get("学历学位", "")).strip() if pd.notna(r.get("学历学位")) else "",
            "专业要求": str(r.get("专业条件要求", "")).strip() if pd.notna(r.get("专业条件要求")) else "",
            "政治面貌": extract_political(other),
            "基层年限": extract_grassroots(other),
            "年龄要求": str(r.get("年龄", "")).strip() if pd.notna(r.get("年龄")) else "",
            "面试比例": str(r.get("面试入围比例", "")).strip() if pd.notna(r.get("面试入围比例")) else "",
            "数据来源": source_file,
            "备注": str(r.get("备注", "")).strip() if pd.notna(r.get("备注")) else "",
        })
    return pd.DataFrame(rows)


def main():
    # ---------- 1. 市州事业单位 ----------
    df_city = pd.read_excel(FILE_CITY, sheet_name="Sheet1", header=None, skiprows=3)
    df_city = df_city.rename(columns={
        0: "市州名称", 1: "县市区", 2: "招聘单位", 3: "岗位类别", 4: "岗位名称",
        5: "岗位编码", 6: "招聘人数", 7: "招聘对象范围", 8: "年龄", 9: "学历学位",
        10: "专业条件要求", 11: "其他", 12: "笔试开考比例", 13: "公共科目笔试名称",
        14: "专业笔试名称", 15: "面试入围比例", 16: "备注", 17: "咨询电话",
    })
    df_city["市州名称"] = df_city["市州名称"].ffill()
    df_city["县市区"] = df_city["县市区"].ffill()
    df_city = df_city[df_city["岗位编码"].notna()]

    # ---------- 2. 省属事业单位 ----------
    df_prov = pd.read_excel(FILE_PROV, sheet_name="Sheet1", header=None, skiprows=3)
    df_prov = df_prov.rename(columns={
        0: "主管部门", 1: "招聘单位", 2: "岗位类别", 3: "岗位名称",
        4: "岗位编码", 5: "招聘人数", 6: "招聘对象范围", 7: "年龄", 8: "学历学位",
        9: "专业条件要求", 10: "其他", 11: "笔试开考比例", 12: "公共科目笔试名称",
        13: "专业笔试名称", 14: "面试入围比例", 15: "备注", 16: "咨询电话",
    })
    df_prov["主管部门"] = df_prov["主管部门"].ffill()
    df_prov = df_prov[df_prov["岗位编码"].notna()]

    # ---------- 3. 省属中小学教师 ----------
    df_teacher = pd.read_excel(FILE_TEACHER, sheet_name="Sheet0", header=1)
    df_teacher.columns = [str(c).strip() for c in df_teacher.columns]
    df_teacher = df_teacher.rename(columns={
        "主管部门": "主管部门",
        "岗位编码": "岗位编码",
        "招聘单位名称": "招聘单位",
        "岗位名称": "岗位名称",
        "岗位类别": "岗位类别",
        "招聘人数": "招聘人数",
        "招聘对象范围": "招聘对象范围",
        "年龄": "年龄",
        "学历学位": "学历学位",
        "专业条件要求": "专业条件要求",
        "笔试开考比例": "笔试开考比例",
        "笔试公共科目": "公共科目笔试名称",
        "笔试专业科目": "专业笔试名称",
        "其他条件": "其他",
        "备注": "备注",
        "咨询电话": "咨询电话",
    })
    df_teacher["主管部门"] = df_teacher["主管部门"].ffill()
    df_teacher = df_teacher[df_teacher["岗位编码"].notna()]

    # ---------- 合并 ----------
    df_city_std = to_standard(df_city, "市州事业单位", FILE_CITY.name, "市州名称")
    df_prov_std = to_standard(df_prov, "省属事业单位", FILE_PROV.name, "主管部门")
    df_teacher_std = to_standard(df_teacher, "省属中小学教师", FILE_TEACHER.name, "主管部门")

    all_df = pd.concat([df_city_std, df_prov_std, df_teacher_std], ignore_index=True)
    out_path = BASE / "data" / "kaogong_data.csv"
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"生成 {len(all_df)} 条考公数据 -> {out_path}")
    print(all_df.head())


if __name__ == "__main__":
    main()