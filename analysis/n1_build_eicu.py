# -*- coding: utf-8 -*-
"""n1_build_eicu.py —— eICU 侧 N1 数据集（landmark 设计，与 MIMIC 对齐）"""
import os
import numpy as np, pandas as pd

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
LM, H = 1440.0, 10080.0

e = pd.read_csv(r"D:/SEEDANCE/EICU/csv/eicu_s3.csv", low_memory=False)
e = e[e["cr_base24"].notna()].reset_index(drop=True).copy()
lab = pd.read_csv(os.path.join(OUT, "n1_eicu_labs.csv"), low_memory=False)
assert len(lab) == len(e) and lab["patientunitstayid"].is_unique
d = e.merge(lab, on="patientunitstayid", how="left", validate="one_to_one")
assert len(d) == len(e)

t0 = pd.to_numeric(d["t0"], errors="coerce").values
dis_off = pd.to_numeric(d["hospitaldischargeoffset"], errors="coerce").values
fu = np.clip(dis_off - t0, None, H)
fu = np.where(np.isnan(fu) | (fu <= 0), H, fu)
aki = pd.to_numeric(d["aki_off"], errors="coerce").values
died = (d["hospitaldischargestatus"].astype(str).str.strip().str.lower() == "expired").values
dt_death = np.where(died, fu, np.nan)

d["aki_pre_lm"] = (~np.isnan(aki)) & (aki <= LM)
d["death_pre_lm"] = died & (fu <= LM)
d["fu_pre_lm"] = fu <= LM
d["eligible_lm"] = ~(d["aki_pre_lm"] | d["death_pre_lm"] | d["fu_pre_lm"])

T = np.where(fu <= LM, np.nan, np.minimum(fu, H))
st = np.zeros(len(d), int)
sel_d = died & (fu > LM)
T = np.where(sel_d, np.minimum(fu, H), T); st = np.where(sel_d, 2, st)
sel_a = (~np.isnan(aki)) & (aki > LM) & (aki <= H) & (aki < np.where(np.isnan(T), np.inf, T))
T = np.where(sel_a, aki, T); st = np.where(sel_a, 1, st)
d["T_lm"] = T - LM
d["status_lm"] = st
d["aki_lm7d"] = (st == 1).astype(int)
d.loc[~d["eligible_lm"], ["T_lm", "status_lm", "aki_lm7d"]] = np.nan

d["bun_cr_ratio"] = d["bun_post_max"] / pd.to_numeric(d["cr_base24"], errors="coerce").replace(0, np.nan)

# 合并有创（vitalperiodic）与无创（vitalaperiodic）血压：取两者最小值（最差值）
for a, b, out in [("sbp_post_min", "sbpn_post_min", "sbp_post_min"),
                  ("dbp_post_min", "dbpn_post_min", "dbp_post_min"),
                  ("mbp_post_min", "mbpn_post_min", "mbp_post_min")]:
    if a in d.columns and b in d.columns:
        d[out] = np.fmin(pd.to_numeric(d[a], errors="coerce"),
                         pd.to_numeric(d[b], errors="coerce"))
d["shock_index_max"] = d["hr_post_max"] / d["sbp_post_min"].replace(0, np.nan)

BASE = {"age": "年龄", "gender": "性别", "sofa_day1": "SOFA(入室首日)", "cr_base24": "基线肌酐(24h 最低)",
        "vent_adm": "机械通气", "vaso": "血管活性药", "xoi_base": "基线 XOI",
        "htn": "高血压", "dm": "糖尿病", "chf": "心衰", "ckd": "慢性肾病", "lung": "慢性肺病",
        "mi": "心梗", "cva": "脑卒中", "cancer": "恶性肿瘤", "liver": "肝病", "gout": "痛风"}
CAND = {"bun_post_max": "BUN 最高", "wbc_post_max": "WBC 最高", "hct_post_min": "Hct 最低",
        "hb_post_min": "Hgb 最低", "plt_post_min": "血小板最低", "rdw_post_max": "RDW 最高",
        "mcv_post_max": "MCV 最高", "mch_post_max": "MCH 最高", "mchc_post_max": "MCHC 最高",
        "na_post_min": "钠最低", "k_post_max": "钾最高", "cl_post_min": "氯最低",
        "ca_post_min": "钙最低", "hco3_post_min": "碳酸氢盐最低", "agap_post_max": "阴离子间隙最高",
        "glu_post_max": "血糖最高", "lac_post_max": "乳酸最高", "ph_post_min": "pH 最低",
        "po2_post_min": "PaO2 最低", "pco2_post_max": "PaCO2 最高", "fio2_post_max": "FiO2 最高",
        "alb_post_min": "白蛋白最低", "tp_post_min": "总蛋白最低",
        "hr_post_max": "心率最高", "sbp_post_min": "收缩压最低", "dbp_post_min": "舒张压最低",
        "mbp_post_min": "平均动脉压最低", "rr_post_max": "呼吸频率最高", "temp_post_max": "体温最高",
        "spo2_post_min": "SpO2 最低", "bun_cr_ratio": "BUN/基线肌酐比", "shock_index_max": "休克指数"}
COUNTS = {"n_lab_post": "实验室记录数", "n_vit_post": "生命体征记录数"}

sub = d[d["eligible_lm"]]
rows = []
for c, name in {**BASE, **CAND, **COUNTS}.items():
    if c not in d.columns: continue
    rows.append(dict(var=c, label=name, kind=("基线" if c in BASE else ("计数" if c in COUNTS else "窗口")),
                     miss_all_pct=round(100 * pd.to_numeric(d[c], errors="coerce").isna().mean(), 1),
                     miss_eligible_pct=round(100 * pd.to_numeric(sub[c], errors="coerce").isna().mean(), 1),
                     median_all=round(float(pd.to_numeric(d[c], errors="coerce").median()), 3)))
ms = pd.DataFrame(rows).sort_values(["kind", "miss_eligible_pct"])
ms.to_csv(os.path.join(OUT, "n1_missingness_eicu.csv"), index=False, encoding="utf-8-sig")

keep = ["patientunitstayid", "eligible_lm", "aki_pre_lm", "death_pre_lm", "fu_pre_lm", "T_lm", "status_lm",
        "aki_lm7d", "followup_lm_min"] if False else None
keep = ["patientunitstayid", "eligible_lm", "aki_pre_lm", "death_pre_lm", "fu_pre_lm", "T_lm", "status_lm",
        "aki_lm7d"] + list(BASE) + list(CAND) + list(COUNTS)
keep = [c for c in keep if c in d.columns]
d[keep].to_csv(os.path.join(OUT, "n1_eicu_dataset.csv"), index=False, encoding="utf-8-sig")

print("总队列 %d；landmark 可入组 %d（%.1f%%）" % (len(d), int(d["eligible_lm"].sum()), 100 * d["eligible_lm"].mean()))
x = d[d["eligible_lm"]]
print("可入组者结局：AKI %d（%.1f%%）／竞争死亡 %d（%.1f%%）／删失 %d（%.1f%%）"
      % ((x["status_lm"] == 1).sum(), 100 * (x["status_lm"] == 1).mean(),
         (x["status_lm"] == 2).sum(), 100 * (x["status_lm"] == 2).mean(),
         (x["status_lm"] == 0).sum(), 100 * (x["status_lm"] == 0).mean()))
print("\n== eICU 候选预测变量缺失率（可入组人群）==")
for lo, hi, tag in [(0, 20, "可用（<20%）"), (20, 50, "需插补（20-50%）"), (50, 101, "剔除（>50%）")]:
    s2 = ms[(ms["miss_eligible_pct"] >= lo) & (ms["miss_eligible_pct"] < hi) & (ms["kind"] == "窗口")]
    print("  %s: %s" % (tag, "、".join(s2["label"].tolist()) or "无"))
print("\nSAVED: n1_eicu_dataset.csv, n1_missingness_eicu.csv")
