# -*- coding: utf-8 -*-
"""n5_cc_fix.py —— 用原始未插补数据重做完整病例（CC）敏感性分析"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import cn2_survival_lib as SL

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
RS = 123
sel = pd.read_csv(os.path.join(OUT, "n4_selected_variables.csv"))
F = sel.loc[sel["primary_intersection"] == True, "var"].tolist()      # noqa: E712
raw = pd.read_csv(os.path.join(OUT, "n4_mimic_dataset.csv"), low_memory=False)
raw = raw[raw["eligible_lm"] == True].reset_index(drop=True).copy()   # noqa: E712
for a, b in [("anchor_age", "age"), ("sofa_max", "sofa"), ("cr_base24", "cr_base")]:
    if a in raw.columns and b not in raw.columns:
        raw[b] = raw[a]
raw["female"] = (raw["gender"].astype(str).str.strip().str.upper() == "F").astype(int)
raw["y"] = raw["aki_lm7d"].astype(int)
missing = [c for c in F if c not in raw.columns]
print("缺列:", missing or "无")
tr_i, va_i = train_test_split(np.arange(len(raw)), test_size=0.3, random_state=RS, stratify=raw["y"].values)
TRr, VAr = raw.loc[tr_i].reset_index(drop=True), raw.loc[va_i].reset_index(drop=True)

def cc_stats(d, tag):
    m = d[F].notna().all(axis=1)
    n_var_miss = d[F].isna().any(axis=1).sum()
    print("%s：n=%d，完整病例 %d（%.1f%%），至少一项缺失 %d" % (tag, len(d), m.sum(), 100 * m.mean(), n_var_miss))
    return m

m_tr = cc_stats(TRr, "训练集（原始）")
m_va = cc_stats(VAr, "验证集（原始）")
A, B = TRr[m_tr].reset_index(drop=True), VAr[m_va].reset_index(drop=True)
print("逐变量缺失率（训练集，>0 者）：")
mr = (100 * TRr[F].isna().mean()).round(2)
print(mr[mr > 0].sort_values(ascending=False).to_string())

rows = []
m = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist")
m.fit(A[F], A["y"])
p = m.predict_proba(B[F])[:, 1]; y = B["y"].values
mm = SL.metrics(B["T_lm"].values, B["status_lm"].values, p)
rows.append(dict(scenario="完整病例（CC，未插补）", model="XGB", n_train=len(A), n=len(y), events=int(y.sum()),
                 auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), uno_c=mm["C_uno"], oe=mm["O_E_7d"]))
print("\nCC 结果：", rows[-1])
s = pd.read_csv(os.path.join(OUT, "n5_sensitivity.csv"))
s = s[~s["scenario"].str.startswith("完整病例")]
s = pd.concat([pd.DataFrame([dict(scenario=r["scenario"], model=r["model"], n=r["n"], events=r["events"],
                                  auc=r["auc"], brier=r["brier"]) for r in rows]), s], ignore_index=True)
s.to_csv(os.path.join(OUT, "n5_sensitivity.csv"), index=False, encoding="utf-8-sig")
print("\nSAVED: n5_sensitivity.csv（已替换为真实 CC 分析）")
