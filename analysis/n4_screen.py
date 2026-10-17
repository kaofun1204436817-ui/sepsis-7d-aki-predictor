# -*- coding: utf-8 -*-
"""n4_screen.py —— N4 扩展池切分/MICE/单因素（流程与 N2 完全一致，仅换变量池）"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import statsmodels.api as sm

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
RS = 123

WIN = ["bun_post_max", "wbc_post_max", "hct_post_min", "hb_post_min", "plt_post_min", "mcv_post_max",
       "mch_post_max", "mchc_post_max", "rdw_post_max", "na_post_min", "k_post_max", "cl_post_min",
       "ca_post_min", "hco3_post_min", "agap_post_max", "glu_post_max", "hr_post_max", "rr_post_max",
       "spo2_post_min", "sbp_post_min", "dbp_post_min", "mbp_post_min", "bun_cr_ratio", "shock_index_max"]
DELTA = ["agap_delta", "hco3_delta", "bun_delta", "ca_delta", "cl_delta", "cr_delta", "glu_delta",
         "na_delta", "k_delta", "hct_delta", "hb_delta", "mch_delta", "mchc_delta", "mcv_delta",
         "plt_delta", "rdw_delta", "hr_delta", "sbp_delta", "dbp_delta", "mbp_delta", "rr_delta",
         "spo2_delta", "cr_ratio", "cr_rise", "bun_cr_delta", "sbp_rel_delta"]
BASE = ["age", "female", "sofa", "cr_base", "vent", "vaso", "xoi_base",
        "htn", "dm", "chf", "ckd", "lung", "mi", "cva", "cancer", "liver", "gout"]
POOL = WIN + DELTA + BASE
CONT = [c for c in POOL if c not in ("female", "vent", "vaso", "xoi_base", "htn", "dm", "chf", "ckd",
                                     "lung", "mi", "cva", "cancer", "liver", "gout")]

def harmonized(path, db):
    d = pd.read_csv(path, low_memory=False)
    d = d[d["eligible_lm"] == True].reset_index(drop=True).copy()   # noqa: E712
    if db == "mimic":
        for a, b in [("anchor_age", "age"), ("sofa_max", "sofa"), ("cr_base24", "cr_base")]:
            if a in d.columns and b not in d.columns:
                d[b] = d[a]
        d["female"] = (d["gender"].astype(str).str.strip().str.upper() == "F").astype(int)
    else:
        for a, b in [("sofa_day1", "sofa"), ("cr_base24", "cr_base"), ("vent_adm", "vent")]:
            if a in d.columns and b not in d.columns:
                d[b] = d[a]
        d["age"] = pd.to_numeric(d["age"].astype(str).str.replace("> 89", "89", regex=False), errors="coerce")
        d["female"] = (d["gender"].astype(str).str.strip().str.upper() == "FEMALE").astype(int)
    d["y"] = d["aki_lm7d"].astype(int)
    avail = [c for c in POOL if c in d.columns]
    for c in avail:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d, avail

m, av_m = harmonized(os.path.join(OUT, "n4_mimic_dataset.csv"), "mimic")
e, av_e = harmonized(os.path.join(OUT, "n4_eicu_dataset.csv"), "eicu")
AVAIL = [c for c in POOL if (c in av_m and c in av_e)]
CONT = [c for c in CONT if c in AVAIL]
print("候选池：设计 %d 个 → 两库均可得 %d 个（连续 %d 个）" % (len(POOL), len(AVAIL), len(CONT)))
print("MIMIC n=%d (AKI %d, %.1f%%) | eICU n=%d (AKI %d, %.1f%%)"
      % (len(m), m.y.sum(), 100 * m.y.mean(), len(e), e.y.sum(), 100 * e.y.mean()))

tr_i, va_i = train_test_split(np.arange(len(m)), test_size=0.3, random_state=RS, stratify=m["y"].values)
imp = IterativeImputer(max_iter=20, random_state=RS, keep_empty_features=True)
imp.fit(m.loc[tr_i, CONT]); med = m.loc[tr_i, CONT].median()

def prep(frame, idx=None, name=""):
    d = (frame if idx is None else frame.loc[idx]).reset_index(drop=True).copy()
    src = d[CONT].copy()
    filled = pd.DataFrame(imp.transform(src), columns=CONT)
    for c in CONT:
        filled[c] = filled[c].fillna(med[c])
    d[CONT] = filled
    for c in CONT:
        if m.loc[tr_i, c].isna().mean() >= 0.02:
            d["miss_" + c] = src[c].isna().astype(int)
    print("  [%s] n=%d 残余缺失 %d" % (name, len(d), int(d[CONT].isna().sum().sum())))
    return d

TR, VA, EX = prep(m, tr_i, "训练"), prep(m, va_i, "验证"), prep(e, None, "外部")
for nm, dd in [("n4_train.csv", TR), ("n4_val.csv", VA), ("n4_ext.csv", EX)]:
    dd.to_csv(os.path.join(OUT, nm), index=False, encoding="utf-8-sig")
open(os.path.join(OUT, "n4_feature_pool.txt"), "w").write("\n".join(AVAIL))

rows = []
for c in AVAIL:
    x = TR[c].astype(float)
    if c in CONT:
        sd = x.std(ddof=0); x = (x - x.mean()) / (sd if sd > 0 else 1.0); unit = "每 1 SD"
    else:
        unit = "有 vs 无"
    if x.nunique() < 2:
        continue
    try:
        f = sm.Logit(TR["y"].values, sm.add_constant(x.values)).fit(disp=0)
        b, se, p = f.params[1], f.bse[1], f.pvalues[1]
        rows.append(dict(var=c, unit=unit, OR=float(np.exp(b)), OR_lo=float(np.exp(b - 1.96 * se)),
                         OR_hi=float(np.exp(b + 1.96 * se)), P=float(p)))
    except Exception:
        continue
uni = pd.DataFrame(rows).sort_values("P")
uni["uni_sig"] = uni["P"] < 0.05
uni.to_csv(os.path.join(OUT, "n4_univariate.csv"), index=False, encoding="utf-8-sig")
print("\n单因素：可得 %d 个，显著 %d 个" % (len(uni), int(uni["uni_sig"].sum())))
print("新增派生特征中的显著者：")
print(uni[uni["var"].isin(DELTA)][["var", "unit", "OR", "OR_lo", "OR_hi", "P"]].round(3).to_string(index=False))
print("\nSAVED: n4_train/val/ext.csv / n4_univariate.csv / n4_feature_pool.txt")
