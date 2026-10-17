# -*- coding: utf-8 -*-
"""
n1_build_dataset.py —— N1 数据集构建（MIMIC-IV 侧，landmark 设计）
设计（严格口径，防信息泄漏）：
  预测时点 = t0 + 24 h（landmark）；预测变量取自 [t0, t0+24h] 窗口（最差值）
  排除 landmark 前已发生 AKI 或死亡者（不死时间偏倚）
  结局 = 7 d 内（t0 起算）landmark 之后发生的 AKI；死亡为竞争事件；随访截断沿用主稿 R 口径
产出：n1_mimic_dataset.csv、n1_missingness.csv
"""
import os
import numpy as np, pandas as pd

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
LM = 1440.0      # landmark, minutes
H = 10080.0      # 7 d horizon from t0

m = pd.read_csv(r"D:/SEEDANCE/EICU/csv/mimic_s3b.csv", low_memory=False)
m = m[m["cr_base24"].notna()].reset_index(drop=True).copy()
lab = pd.read_csv(os.path.join(OUT, "n1_mimic_labs.csv"), low_memory=False)
assert lab["stay_id"].is_unique and len(lab) == len(m)
d = m.merge(lab, on="stay_id", how="left", validate="one_to_one")
assert len(d) == len(m)

# ---------------- 时间与结局（以 landmark 为原点） ----------------
t0 = pd.to_datetime(d["t0"], errors="coerce")
dis = pd.to_datetime(d["dischtime"], errors="coerce")
dea = pd.to_datetime(d["deathtime"], errors="coerce")
fu = ((dis - t0).dt.total_seconds() / 60.0).clip(lower=1, upper=H).fillna(H).values      # 随访上限（分钟，自 t0）
aki_off = pd.to_numeric(d["aki_off"], errors="coerce").values
dt_death = ((dea - t0).dt.total_seconds() / 60.0).values
died = (pd.to_numeric(d["hospital_expire_flag"], errors="coerce") == 1).values
death_in = died & (~np.isnan(dt_death)) & (dt_death >= 0) & (dt_death <= fu)

d["aki_pre_lm"] = (~np.isnan(aki_off)) & (aki_off <= LM)
d["death_pre_lm"] = death_in & (~np.isnan(dt_death)) & (dt_death <= LM)
d["fu_pre_lm"] = fu <= LM          # 随访在 landmark 前结束（无可评价观察窗）
d["eligible_lm"] = ~(d["aki_pre_lm"] | d["death_pre_lm"] | d["fu_pre_lm"])

T = np.where(fu <= LM, np.nan, np.minimum(fu, H))          # 观察时长（自 t0），<landmark 者无效
st = np.zeros(len(d), int)
sel_d = death_in & (dt_death > LM)
T = np.where(sel_d, np.minimum(np.where(np.isnan(dt_death), H, dt_death), H), T)
st = np.where(sel_d, 2, st)
sel_a = (~np.isnan(aki_off)) & (aki_off > LM) & (aki_off <= H) & (aki_off < np.where(np.isnan(T), np.inf, T))
T = np.where(sel_a, aki_off, T)
st = np.where(sel_a, 1, st)
d["T_lm"] = T - LM                     # 以 landmark 为原点的观察时长（分钟）
d["status_lm"] = st                    # 1=AKI, 2=死亡(竞争), 0=删失
d["aki_lm7d"] = ((d["status_lm"] == 1)).astype(int)
d.loc[~d["eligible_lm"], ["T_lm", "status_lm", "aki_lm7d"]] = np.nan
d["followup_lm_min"] = np.where(d["eligible_lm"], np.minimum(H - LM, np.maximum(fu - LM, 0)), np.nan)

# ---------------- 候选预测变量（[t0, t0+24h] 最差值） ----------------
CAND = {
    "alb_post_min": "白蛋白最低", "bun_post_max": "BUN 最高", "glo_post_min": "球蛋白最低",
    "tp_post_min": "总蛋白最低", "agap_post_max": "阴离子间隙最高", "hco3_post_min": "碳酸氢盐最低",
    "ca_post_min": "钙最低", "cl_post_min": "氯最低", "glu_post_max": "血糖最高",
    "na_post_min": "钠最低", "k_post_max": "钾最高",
    "hct_post_min": "红细胞压积最低", "hb_post_min": "血红蛋白最低", "mch_post_max": "MCH 最高",
    "mchc_post_max": "MCHC 最高", "mcv_post_max": "MCV 最高", "plt_post_min": "血小板最低",
    "rbc_post_min": "红细胞最低", "rdw_post_max": "RDW 最高", "wbc_post_max": "WBC 最高",
    "lac_post_max": "乳酸最高", "ph_post_min": "pH 最低", "po2_post_min": "PaO2 最低",
    "pco2_post_max": "PaCO2 最高", "fio2_post_max": "FiO2 最高", "pf_post_min": "P/F 比最低",
    "be_post_min": "碱剩余最低", "so2_post_min": "SO2 最低",
    "hr_post_max": "心率最高", "sbp_post_min": "收缩压最低", "dbp_post_min": "舒张压最低",
    "mbp_post_min": "平均动脉压最低", "rr_post_max": "呼吸频率最高", "temp_post_max": "体温最高",
    "spo2_post_min": "SpO2 最低", "gcs_post_min": "GCS 最低",
    "uo6_post_min": "6 h 尿量最低", "uo12_post_min": "12 h 尿量最低", "uo24_post_min": "24 h 尿量最低",
    "n_crrt_pre": "landmark 前 CRRT 记录数",
}
COUNTS = {"n_chem_post": "生化检测次数", "n_cbc_post": "血常规次数", "n_bg_post": "血气次数",
          "n_vit_post": "生命体征次数", "n_gcs_post": "GCS 次数", "n_uo_post": "尿量记录次数"}
d["bun_cr_ratio"] = d["bun_post_max"] / d["cr_base24"].replace(0, np.nan)          # 派生：BUN/Cr
d["shock_index_max"] = d["hr_post_max"] / d["sbp_post_min"].replace(0, np.nan)     # 派生：休克指数
CAND["bun_cr_ratio"] = "BUN/基线肌酐比"; CAND["shock_index_max"] = "休克指数"

# 基线变量（现稿已有，landmark 设计下仍可直接用）
BASE = {"anchor_age": "年龄", "gender": "性别", "sofa_max": "SOFA 最高", "cr_base24": "基线肌酐(24h 最低)",
        "vent": "机械通气", "vaso": "血管活性药", "xoi_base": "基线 XOI",
        "htn": "高血压", "dm": "糖尿病", "chf": "心衰", "ckd": "慢性肾病", "lung": "慢性肺病",
        "mi": "心梗", "cva": "脑卒中", "cancer": "恶性肿瘤", "liver": "肝病", "gout": "痛风"}

# ---------------- 缺失率表 ----------------
rows = []
sub = d[d["eligible_lm"]]
for c, name in {**BASE, **CAND, **COUNTS}.items():
    if c not in d.columns:
        continue
    rows.append(dict(var=c, label=name, kind=("基线" if c in BASE else ("计数" if c in COUNTS else "窗口")),
                     miss_all_pct=round(100 * d[c].isna().mean(), 1),
                     miss_eligible_pct=round(100 * sub[c].isna().mean(), 1),
                     median_all=round(float(pd.to_numeric(d[c], errors="coerce").median()), 3)
                     if pd.api.types.is_numeric_dtype(d[c]) else np.nan))
ms = pd.DataFrame(rows).sort_values(["kind", "miss_eligible_pct"])
ms.to_csv(os.path.join(OUT, "n1_missingness.csv"), index=False, encoding="utf-8-sig")

keep = ["stay_id", "subject_id", "hadm_id", "eligible_lm", "aki_pre_lm", "death_pre_lm", "fu_pre_lm",
        "T_lm", "status_lm", "aki_lm7d", "followup_lm_min"] + list(BASE) + list(CAND) + list(COUNTS)
keep = [c for c in keep if c in d.columns]
d[keep].to_csv(os.path.join(OUT, "n1_mimic_dataset.csv"), index=False, encoding="utf-8-sig")

print("总队列 %d；landmark 可入组 %d（%.1f%%）" % (len(d), int(d["eligible_lm"].sum()), 100 * d["eligible_lm"].mean()))
e = d[d["eligible_lm"]]
print("可入组者结局：AKI %d（%.1f%%）／竞争死亡 %d（%.1f%%）／删失 %d（%.1f%%）"
      % ((e["status_lm"] == 1).sum(), 100 * (e["status_lm"] == 1).mean(),
         (e["status_lm"] == 2).sum(), 100 * (e["status_lm"] == 2).mean(),
         (e["status_lm"] == 0).sum(), 100 * (e["status_lm"] == 0).mean()))
print("\n== 预测变量缺失率（可入组人群，按<20% / 20-50% / >50% 分层）==")
for lo, hi, tag in [(0, 20, "可用（<20%）"), (20, 50, "需插补（20-50%）"), (50, 101, "主模型剔除（>50%）")]:
    sel = ms[(ms["miss_eligible_pct"] >= lo) & (ms["miss_eligible_pct"] < hi) & (ms["kind"] == "窗口")]
    print("  %s: %s" % (tag, "、".join(sel["label"].tolist()) or "无"))
print("\nSAVED: n1_mimic_dataset.csv, n1_missingness.csv")
