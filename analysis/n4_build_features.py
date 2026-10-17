# -*- coding: utf-8 -*-
"""
n4_build_features.py —— N4 第一步：特征工程扩展（时间趋势/派生量）
在 N1 已有 min/max 提取基础上派生 delta（max−min，24 h 窗内变化幅度）与比值类特征，
再与基线变量合并为扩展候选池；缺失率 >20% 者剔除（沿用同一规则）。
产出：n4_mimic_dataset.csv / n4_eicu_dataset.csv / n4_missingness_pool.csv
"""
import os
import numpy as np, pandas as pd

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
LM, H = 1440.0, 10080.0

# 原始 N1 提取中的 min/max 配对（用于派生 delta）
PAIRS_M = [  # (短名, 中文)
    ("alb", "白蛋白"), ("glo", "球蛋白"), ("tp", "总蛋白"), ("agap", "阴离子间隙"),
    ("hco3", "碳酸氢盐"), ("bun", "BUN"), ("ca", "钙"), ("cl", "氯"), ("cr", "肌酐"),
    ("glu", "血糖"), ("na", "钠"), ("k", "钾"), ("hct", "Hct"), ("hb", "Hgb"),
    ("mch", "MCH"), ("mchc", "MCHC"), ("mcv", "MCV"), ("plt", "PLT"), ("rbc", "RBC"),
    ("rdw", "RDW"), ("lac", "乳酸"), ("ph", "pH"), ("po2", "PaO2"), ("pco2", "PaCO2"),
    ("fio2", "FiO2"), ("pf", "P/F 比"), ("be", "碱剩余"), ("so2", "SO2"),
    ("hr", "心率"), ("sbp", "收缩压"), ("dbp", "舒张压"), ("mbp", "平均动脉压"),
    ("rr", "呼吸频率"), ("temp", "体温"), ("spo2", "SpO2"), ("gcs", "GCS"),
    ("uo6", "6 h 尿量"), ("uo12", "12 h 尿量"), ("uo24", "24 h 尿量"),
]
PAIRS_E_EXTRA = [("sbpn", "无创收缩压"), ("dbpn", "无创舒张压"), ("mbpn", "无创平均压")]

def build(db):
    if db == "mimic":
        lab = pd.read_csv(os.path.join(OUT, "n1_mimic_labs.csv"), low_memory=False)
        ds = pd.read_csv(os.path.join(OUT, "n1_mimic_dataset.csv"), low_memory=False)
        pairs = PAIRS_M
        base_map = {"anchor_age": "age", "sofa_max": "sofa", "cr_base24": "cr_base"}
        lab_sel = lab[["stay_id"] + [c for c in lab.columns if c.endswith(("_post_min", "_post_max"))]]
        d = ds.copy()
        key = "stay_id"
        for c in lab_sel.columns:
            if c == key or c in d.columns:
                continue          # ds 已有列（含已融合的血压）优先
            d[c] = d[key].map(lab_sel.set_index(key)[c])
        for a, b in base_map.items():
            if a in d.columns and b not in d.columns:
                d[b] = d[a]
        d["female"] = (d["gender"].astype(str).str.strip().str.upper() == "F").astype(int)
    else:
        lab = pd.read_csv(os.path.join(OUT, "n1_eicu_labs.csv"), low_memory=False)
        ds = pd.read_csv(os.path.join(OUT, "n1_eicu_dataset.csv"), low_memory=False)
        pairs = PAIRS_M + PAIRS_E_EXTRA
        lab_sel = lab[["patientunitstayid"] + [c for c in lab.columns if c.endswith(("_post_min", "_post_max"))]]
        d = ds.copy()
        key = "patientunitstayid"
        for c in lab_sel.columns:
            if c == key or c in d.columns:
                continue          # ds 已有列（含已融合的血压）优先
            d[c] = d[key].map(lab_sel.set_index(key)[c])
        d["age"] = pd.to_numeric(d["age"].astype(str).str.replace("> 89", "89", regex=False), errors="coerce")
        d["sofa"] = pd.to_numeric(d["sofa_day1"], errors="coerce")
        d["cr_base"] = pd.to_numeric(d["cr_base24"], errors="coerce")
        d["vent"] = pd.to_numeric(d.get("vent_adm"), errors="coerce")
        d["female"] = (d["gender"].astype(str).str.strip().str.upper() == "FEMALE").astype(int)

    # eICU：有创＋无创血压双侧融合（min 与 max 各取极值）
    for a, b in [("sbp", "sbpn"), ("dbp", "dbpn"), ("mbp", "mbpn")]:
        for side in ("min", "max"):
            ca, cb = f"{a}_post_{side}", f"{b}_post_{side}"
            if cb in d.columns:
                va = pd.to_numeric(d[ca], errors="coerce") if ca in d.columns else pd.Series(np.nan, index=d.index)
                vb = pd.to_numeric(d[cb], errors="coerce")
                d[ca] = np.fmin(va, vb) if side == "min" else np.fmax(va, vb)

    # 派生 delta（24 h 窗内 max − min）
    made = []
    for s, cn in pairs:
        lo, hi = f"{s}_post_min", f"{s}_post_max"
        if lo in d.columns and hi in d.columns:
            if s in ("sbpn", "dbpn", "mbpn"):
                # 无创血压与有创合并后再算 delta
                base = s.replace("n", "")
                if f"{base}_post_min" in d.columns:
                    mn = np.fmin(pd.to_numeric(d[f"{base}_post_min"], errors="coerce"),
                                 pd.to_numeric(d[lo], errors="coerce"))
                    mx = np.fmax(pd.to_numeric(d[f"{base}_post_max"], errors="coerce"),
                                 pd.to_numeric(d[hi], errors="coerce"))
                    d[f"{s}_delta"] = mx - mn
                    made.append((f"{s}_delta", f"{cn} 变化幅度"))
                continue
            d[f"{s}_delta"] = pd.to_numeric(d[hi], errors="coerce") - pd.to_numeric(d[lo], errors="coerce")
            made.append((f"{s}_delta", f"{cn} 变化幅度"))
    # 比值类
    if "cr_post_max" in d.columns:
        d["cr_ratio"] = pd.to_numeric(d["cr_post_max"], errors="coerce") / \
            pd.to_numeric(d["cr_base"], errors="coerce").replace(0, np.nan)
        made.append(("cr_ratio", "肌酐峰值/基线比"))
        d["cr_rise"] = pd.to_numeric(d["cr_post_max"], errors="coerce") - pd.to_numeric(d["cr_base"], errors="coerce")
        made.append(("cr_rise", "肌酐升高绝对值"))
    if "bun_post_max" in d.columns and "cr_post_max" in d.columns:
        d["bun_cr_delta"] = pd.to_numeric(d["bun_post_max"], errors="coerce") / \
            pd.to_numeric(d["cr_post_max"], errors="coerce").replace(0, np.nan)
        made.append(("bun_cr_delta", "BUN/肌酐峰值比"))
    # 归一化血压变化
    if "sbp_delta" in d.columns:
        d["sbp_rel_delta"] = d["sbp_delta"] / pd.to_numeric(d["sbp_post_max"], errors="coerce").replace(0, np.nan)
        made.append(("sbp_rel_delta", "收缩压相对波动"))
    return d, made

m, made_m = build("mimic")
e, made_e = build("eicu")
print("MIMIC 派生特征 %d 个；eICU 派生特征 %d 个" % (len(made_m), len(made_e)))
print("示例：", "、".join(x[1] for x in made_m[:8]), "...")

# 保存
m.to_csv(os.path.join(OUT, "n4_mimic_dataset.csv"), index=False, encoding="utf-8-sig")
e.to_csv(os.path.join(OUT, "n4_eicu_dataset.csv"), index=False, encoding="utf-8-sig")

# 缺失率清单（派生特征，可入组人群）
elig_m = pd.to_numeric(m["eligible_lm"], errors="coerce") == 1
elig_e = pd.to_numeric(e["eligible_lm"], errors="coerce") == 1
rows = []
for name, cn in made_m:
    rows.append(dict(feature=name, label=cn,
                     mimic_pct=round(100 * pd.to_numeric(m.loc[elig_m, name], errors="coerce").isna().mean(), 1),
                     eicu_pct=(round(100 * pd.to_numeric(e.loc[elig_e, name], errors="coerce").isna().mean(), 1)
                               if name in e.columns else np.nan)))
ms = pd.DataFrame(rows)
ms["usable_both"] = (ms["mimic_pct"] < 20) & (ms["eicu_pct"].fillna(999) < 20)
ms.sort_values(["usable_both", "mimic_pct"], ascending=[False, True]).to_csv(
    os.path.join(OUT, "n4_missingness_pool.csv"), index=False, encoding="utf-8-sig")
print("\n两库均可用（缺失率<20%%）派生特征 %d 个：" % int(ms["usable_both"].sum()))
print("、".join(ms.loc[ms["usable_both"], "label"].tolist()))
print("\n仅 MIMIC 可用（eICU 缺失或 >20%）：")
print("、".join(ms.loc[~ms["usable_both"], "label"].tolist()))
print("\nSAVED: n4_mimic_dataset.csv / n4_eicu_dataset.csv / n4_missingness_pool.csv")
