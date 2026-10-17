# -*- coding: utf-8 -*-
"""
n2_prepare.py v2 —— N2 第一步（修正版）
修正：① 训练/验证/外部严格子集切分（v1 误返回全量）；② MICE 残差 NaN 以训练集中位数兜底并计数报告；
      ③ 单因素拟合异常显式打印，不再静默 NaN。
产出：n2_train.csv / n2_val.csv / n2_ext.csv、n2_univariate.csv、n2_univariate.png
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import statsmodels.api as sm

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
RS = 123

WIN = ["bun_post_max", "wbc_post_max", "hct_post_min", "hb_post_min", "plt_post_min",
       "mcv_post_max", "mch_post_max", "mchc_post_max", "rdw_post_max", "na_post_min",
       "k_post_max", "cl_post_min", "ca_post_min", "hco3_post_min", "agap_post_max",
       "glu_post_max", "hr_post_max", "rr_post_max", "spo2_post_min", "sbp_post_min",
       "dbp_post_min", "mbp_post_min", "bun_cr_ratio", "shock_index_max"]
BASE = ["age", "female", "sofa", "cr_base", "vent", "vaso", "xoi_base",
        "htn", "dm", "chf", "ckd", "lung", "mi", "cva", "cancer", "liver", "gout"]
FEATS = WIN + BASE
CONT = ["bun_post_max", "wbc_post_max", "hct_post_min", "hb_post_min", "plt_post_min", "mcv_post_max",
        "mch_post_max", "mchc_post_max", "rdw_post_max", "na_post_min", "k_post_max", "cl_post_min",
        "ca_post_min", "hco3_post_min", "agap_post_max", "glu_post_max", "hr_post_max", "rr_post_max",
        "spo2_post_min", "sbp_post_min", "dbp_post_min", "mbp_post_min", "bun_cr_ratio",
        "shock_index_max", "age", "sofa", "cr_base"]
LABELS = {"bun_post_max": "BUN 最高", "wbc_post_max": "WBC 最高", "hct_post_min": "Hct 最低",
          "hb_post_min": "Hgb 最低", "plt_post_min": "PLT 最低", "mcv_post_max": "MCV 最高",
          "mch_post_max": "MCH 最高", "mchc_post_max": "MCHC 最高", "rdw_post_max": "RDW 最高",
          "na_post_min": "Na 最低", "k_post_max": "K 最高", "cl_post_min": "Cl 最低",
          "ca_post_min": "Ca 最低", "hco3_post_min": "HCO3 最低", "agap_post_max": "阴离子间隙最高",
          "glu_post_max": "血糖最高", "hr_post_max": "HR 最高", "rr_post_max": "RR 最高",
          "spo2_post_min": "SpO2 最低", "sbp_post_min": "SBP 最低", "dbp_post_min": "DBP 最低",
          "mbp_post_min": "MAP 最低", "bun_cr_ratio": "BUN/Cr 比", "shock_index_max": "休克指数",
          "age": "年龄", "female": "女性", "sofa": "SOFA", "cr_base": "基线肌酐", "vent": "机械通气",
          "vaso": "血管活性药", "xoi_base": "基线 XOI", "htn": "高血压", "dm": "糖尿病",
          "chf": "心衰", "ckd": "慢性肾病", "lung": "慢性肺病", "mi": "心梗", "cva": "脑卒中",
          "cancer": "恶性肿瘤", "liver": "肝病", "gout": "痛风"}

def harmonized(path, db):
    d = pd.read_csv(path, low_memory=False)
    d = d[d["eligible_lm"] == True].reset_index(drop=True).copy()   # noqa: E712
    if db == "mimic":
        d = d.rename(columns={"anchor_age": "age", "sofa_max": "sofa", "cr_base24": "cr_base"})
        d["female"] = (d["gender"].astype(str).str.strip().str.upper() == "F").astype(int)
    else:
        d = d.rename(columns={"sofa_day1": "sofa", "cr_base24": "cr_base", "vent_adm": "vent"})
        d["age"] = pd.to_numeric(d["age"].astype(str).str.replace("> 89", "89", regex=False), errors="coerce")
        d["female"] = (d["gender"].astype(str).str.strip().str.upper() == "FEMALE").astype(int)
    d["y"] = d["aki_lm7d"].astype(int)
    for c in WIN + BASE:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[FEATS + ["y"]].copy()

m = harmonized(os.path.join(OUT, "n1_mimic_dataset.csv"), "mimic")
e = harmonized(os.path.join(OUT, "n1_eicu_dataset.csv"), "eicu")
print("MIMIC landmark n=%d (AKI %d, %.1f%%) | eICU n=%d (AKI %d, %.1f%%)"
      % (len(m), m.y.sum(), 100 * m.y.mean(), len(e), e.y.sum(), 100 * e.y.mean()))

tr_i, va_i = train_test_split(np.arange(len(m)), test_size=0.3, random_state=RS, stratify=m["y"].values)
print("切分：训练 %d（AKI %.1f%%）／验证 %d（AKI %.1f%%）"
      % (len(tr_i), 100 * m["y"].values[tr_i].mean(), len(va_i), 100 * m["y"].values[va_i].mean()))

# ---------------- MICE（仅训练集拟合）＋中位数兜底 ----------------
imp = IterativeImputer(max_iter=20, random_state=RS, keep_empty_features=True)
imp.fit(m.loc[tr_i, CONT])
med = m.loc[tr_i, CONT].median()
miss_floor = {c for c in CONT if m.loc[tr_i, c].isna().mean() >= 0.02}

def make(frame, idx=None, name=""):
    d = (frame if idx is None else frame.loc[idx]).copy().reset_index(drop=True)
    src = d[CONT].copy()
    filled = imp.transform(src)
    n_res = int(np.isnan(filled).sum())
    filled = pd.DataFrame(filled, columns=CONT)
    # 残差兜底：训练集中位数
    for c in CONT:
        filled[c] = filled[c].fillna(med[c])
    if n_res:
        print("  [%s] MICE 残差 NaN %d 个 → 训练集中位数兜底" % (name, n_res))
    d[CONT] = filled
    for c in sorted(miss_floor):
        d["miss_" + c] = src[c].isna().astype(int)
    left = int(d[CONT].isna().sum().sum())
    print("  [%s] n=%d  插补后连续变量残余缺失 %d" % (name, len(d), left))
    return d

tr = make(m, tr_i, "训练")
va = make(m, va_i, "验证")
ex = make(e, None, "外部")
MISSF = [c for c in tr.columns if c.startswith("miss_")]
print("缺失指示变量 %d 个：%s" % (len(MISSF), "、".join(c[5:] for c in MISSF) or "无"))
for nm, d in [("n2_train.csv", tr), ("n2_val.csv", va), ("n2_ext.csv", ex)]:
    d.to_csv(os.path.join(OUT, nm), index=False, encoding="utf-8-sig")

# ---------------- 单因素 logistic（仅训练集） ----------------
rows, bad = [], []
for c in FEATS:
    x = tr[c].astype(float)
    if c in CONT:
        sd = x.std(ddof=0); x = (x - x.mean()) / (sd if sd > 0 else 1.0); unit = "每 1 SD"
    else:
        unit = "有 vs 无"
    if x.notna().sum() < len(x) or x.nunique() < 2:
        bad.append("%s(非空%d/取值%d)" % (c, int(x.notna().sum()), int(x.nunique())))
        continue
    try:
        f = sm.Logit(tr["y"].values, sm.add_constant(x.values)).fit(disp=0)
        b, se, p = f.params[1], f.bse[1], f.pvalues[1]
        rows.append(dict(var=c, label=LABELS.get(c, c), unit=unit, OR=float(np.exp(b)),
                         OR_lo=float(np.exp(b - 1.96 * se)), OR_hi=float(np.exp(b + 1.96 * se)),
                         beta=float(b), P=float(p)))
    except Exception as ex_:
        bad.append("%s(%s)" % (c, type(ex_).__name__))
uni = pd.DataFrame(rows).sort_values("P")
uni["uni_sig"] = uni["P"] < 0.05
uni.to_csv(os.path.join(OUT, "n2_univariate.csv"), index=False, encoding="utf-8-sig")
print("\n单因素完成：%d/%d 变量可得；异常/跳过：%s" % (len(uni), len(FEATS), "、".join(bad) or "无"))
print("显著 (P<0.05)：%d 个" % int(uni["uni_sig"].sum()))
print(uni[["label", "unit", "OR", "OR_lo", "OR_hi", "P"]].round(3).to_string(index=False))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    u = uni.dropna(subset=["OR"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, max(4, 0.26 * len(u))))
    y = np.arange(len(u))
    ax.hlines(y, u["OR_lo"], u["OR_hi"], color="#95a5a6", lw=1)
    ax.scatter(u["OR"], y, c=["#c0392b" if s else "#7f8c8d" for s in u["uni_sig"]], s=24, zorder=3)
    ax.axvline(1, ls="--", c="#34495e", lw=0.9)
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(u["label"], fontsize=8)
    ax.set_xlabel("单因素 OR（95%CI，对数刻度）")
    ax.set_title("N2 单因素筛选：landmark 后 7 d AKI 预测变量（MIMIC 训练集 n=%d）" % len(tr), fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "n2_univariate.png"), dpi=300)
    print("\nSAVED: n2_univariate.png")
except Exception as ex_:
    print("plot skipped:", ex_)
