# -*- coding: utf-8 -*-
"""
n3_models.py —— N3：8 模型矩阵 + GridSearchCV 调参 + 三集三性评价
变量集：N2 三方法交集 18 变量（主）；敏感性：+XOI（19 变量）、LASSO λ.min（37 变量）
评价：AUC+95%CI、DeLong、PR-AUC、混淆矩阵全指标、校准（Brier/HL/斜率/截距/十分位）、
      DCA、风险四分位、竞争风险指标（Uno's C、排除死亡 AUC、时间依赖 AUC、IPCW Brier）
产出：n3_*.csv / n3_*.png / n3_best_model.pkl
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix, precision_score,
                             recall_score, f1_score, cohen_kappa_score, brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import cn2_survival_lib as SL

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
RS = 123
t_start = time.time()
def log(*a): print("[%6.1f min] " % ((time.time() - t_start) / 60) + " ".join(str(x) for x in a), flush=True)

WIN = ["bun_post_max", "wbc_post_max", "hct_post_min", "hb_post_min", "plt_post_min", "mcv_post_max",
       "mch_post_max", "mchc_post_max", "rdw_post_max", "na_post_min", "k_post_max", "cl_post_min",
       "ca_post_min", "hco3_post_min", "agap_post_max", "glu_post_max", "hr_post_max", "rr_post_max",
       "spo2_post_min", "sbp_post_min", "dbp_post_min", "mbp_post_min", "bun_cr_ratio", "shock_index_max"]
BASE = ["age", "female", "sofa", "cr_base", "vent", "vaso", "xoi_base",
        "htn", "dm", "chf", "ckd", "lung", "mi", "cva", "cancer", "liver", "gout"]
CONT = ["bun_post_max", "wbc_post_max", "hct_post_min", "hb_post_min", "plt_post_min", "mcv_post_max",
        "mch_post_max", "mchc_post_max", "rdw_post_max", "na_post_min", "k_post_max", "cl_post_min",
        "ca_post_min", "hco3_post_min", "agap_post_max", "glu_post_max", "hr_post_max", "rr_post_max",
        "spo2_post_min", "sbp_post_min", "dbp_post_min", "mbp_post_min", "bun_cr_ratio",
        "shock_index_max", "age", "sofa", "cr_base"]

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
    return d

m = harmonized(os.path.join(OUT, "n1_mimic_dataset.csv"), "mimic")
e = harmonized(os.path.join(OUT, "n1_eicu_dataset.csv"), "eicu")
tr_i, va_i = train_test_split(np.arange(len(m)), test_size=0.3, random_state=RS, stratify=m["y"].values)

imp = IterativeImputer(max_iter=20, random_state=RS, keep_empty_features=True)
imp.fit(m.loc[tr_i, CONT]); med = m.loc[tr_i, CONT].median()

def prep(frame, idx=None):
    d = (frame if idx is None else frame.loc[idx]).reset_index(drop=True).copy()
    src = d[CONT].copy()
    filled = pd.DataFrame(imp.transform(src), columns=CONT)
    for c in CONT:
        filled[c] = filled[c].fillna(med[c])
    d[CONT] = filled
    for c in CONT:
        if m.loc[tr_i, c].isna().mean() >= 0.02:
            d["miss_" + c] = src[c].isna().astype(int)
    return d

TR, VA, EX = prep(m, tr_i), prep(m, va_i), prep(e, None)
sel = pd.read_csv(os.path.join(OUT, "n2_selected_variables.csv"))
F18 = sel.loc[sel["primary_intersection"] == True, "var"].tolist()      # noqa: E712
las = pd.read_csv(os.path.join(OUT, "n2_lasso.csv"))
F37 = las.loc[las["lasso_min"] == True, "var"].tolist()                 # noqa: E712
F19 = F18 + ["xoi_base"]
log("变量集：主 %d 个 / +XOI %d 个 / λ.min %d 个" % (len(F18), len(F19), len(F37)))

# ---------------------------------------------------------------- 指标函数
def delong_pairs(y, preds):
    """成对 DeLong 检验（AUC 比较）。"""
    from scipy.stats import norm
    def midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N); T2[J] = T
        return T2
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    names = list(preds)
    V = {}
    for k in names:
        x = preds[k]
        tx = midrank(x); ty = midrank(y)
        tz = midrank(x[y == 0]); txx = midrank(x[y == 1])
        auc = (tx[y == 1].sum() / n1 - (n1 + 1) / 2.0) / n0
        v01 = (tx[y == 0] - tz) / n0
        v10 = (tx[y == 1] - txx) / n1
        V[k] = (auc, v01, v10)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            auc1, auc2 = V[a][0], V[b][0]
            s01 = np.cov(np.vstack([V[a][1], V[b][1]]))[0, 1]
            s10 = np.cov(np.vstack([V[a][2], V[b][2]]))[0, 1]
            var = (s01 / n0) + (s10 / n1)
            z = (auc1 - auc2) / np.sqrt(var) if var > 0 else 0.0
            p = 2 * (1 - norm.cdf(abs(z)))
            out.append(dict(model_1=a, model_2=b, auc_1=auc1, auc_2=auc2, z=z, P=p))
    return pd.DataFrame(out)

def hl_test(y, p, g=10):
    q = pd.qcut(pd.Series(p), g, labels=False, duplicates="drop")
    obs1 = pd.Series(y).groupby(q).sum(); exp1 = pd.Series(p).groupby(q).sum()
    n_g = pd.Series(y).groupby(q).size()
    obs0 = n_g - obs1; exp0 = n_g - exp1
    stat = float((((obs1 - exp1) ** 2) / np.maximum(exp1, 1e-9) + ((obs0 - exp0) ** 2) / np.maximum(exp0, 1e-9)).sum())
    from scipy.stats import chi2
    dfree = max(len(n_g) - 2, 1)
    return stat, dfree, float(1 - chi2.cdf(stat, dfree))

def calib_slope(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    import statsmodels.api as sm
    try:
        f = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
        return float(f.params[1]), float(f.params[0])
    except Exception:
        return np.nan, np.nan

def youden_thr(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])

def evaluate(T_all, S_all, y, p, tag):
    """一次算全部指标；T_all/S_all 为 landmark 原点的时间与状态。"""
    auc = roc_auc_score(y, p)
    bs = brier_score_loss(y, p)
    thr = youden_thr(y, p); yh = (p >= thr).astype(int)
    cm = confusion_matrix(y, yh)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
    hl, hldf, hlp = hl_test(y, p)
    slope, inter = calib_slope(y, p)
    m = SL.metrics(T_all, S_all, p)
    return dict(tag=tag, n=len(y), events=int(y.sum()), auc=auc, brier=bs,
                pr_auc=average_precision_score(y, p), thr_youden=thr,
                sens=tp / (tp + fn) if (tp + fn) else np.nan,
                spec=tn / (tn + fp) if (tn + fp) else np.nan,
                ppv=tp / (tp + fp) if (tp + fp) else np.nan,
                npv=tn / (tn + fn) if (tn + fn) else np.nan,
                f1=f1_score(y, yh), kappa=cohen_kappa_score(y, yh),
                hl_stat=hl, hl_df=hldf, hl_P=hlp, cal_slope=slope, cal_int=inter,
                uno_c=m["C_uno"], auc_exd=m["AUC_exd"], brier_exd=m["Brier_exd"],
                auc_t1=m["AUC_t1"], auc_t3=m["AUC_t3"], auc_t6=m["AUC_t6"], oe=m["O_E_7d"])

def boot_ci(T_all, S_all, y, p, B=300, seed=20261016):
    rng = np.random.default_rng(seed)
    aucs, briers, unos = [], [], []
    for _ in range(B):
        bi = rng.integers(0, len(y), len(y))
        if y[bi].sum() == 0 or y[bi].sum() == len(bi):
            continue
        aucs.append(roc_auc_score(y[bi], p[bi])); briers.append(brier_score_loss(y[bi], p[bi]))
        try:
            unos.append(SL.metrics(T_all[bi], S_all[bi], p[bi], with_binary=False)["C_uno"])
        except Exception:
            pass
    return (np.nanpercentile(aucs, [2.5, 97.5]), np.nanpercentile(briers, [2.5, 97.5]),
            np.nanpercentile(unos, [2.5, 97.5]) if unos else [np.nan, np.nan])

# ---------------------------------------------------------------- 模型矩阵
def z(): return StandardScaler()
MODELS = {
    "LR": (Pipeline([("sc", z()), ("m", LogisticRegression(max_iter=3000, random_state=RS))]),
           {"m__C": [0.01, 0.1, 1.0, 10.0], "m__penalty": ["l2"]}),
    "RF": (RandomForestClassifier(random_state=RS, n_jobs=-1),
           {"n_estimators": [300, 600], "max_depth": [None, 12, 20], "min_samples_leaf": [1, 5]}),
    "XGB": (XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist"),
            {"n_estimators": [300, 600], "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1],
             "subsample": [0.8], "colsample_bytree": [0.8]}),
    "LGBM": (LGBMClassifier(random_state=RS, n_jobs=-1, verbose=-1),
             {"n_estimators": [300, 600], "num_leaves": [31, 63], "learning_rate": [0.03, 0.1]}),
    "SVM": (Pipeline([("sc", z()), ("m", SVC(probability=True, random_state=RS, cache_size=1024))]),
            {"m__C": [0.5, 2.0], "m__gamma": ["scale"]}),
    "MLP": (Pipeline([("sc", z()), ("m", MLPClassifier(random_state=RS, max_iter=300, early_stopping=True))]),
            {"m__hidden_layer_sizes": [(64,), (128, 64)], "m__alpha": [1e-3, 1e-2]}),
    "NB": (GaussianNB(), {"var_smoothing": [1e-9, 1e-7]}),
    "BAG": (BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS),
                              random_state=RS, n_jobs=-1),
            {"n_estimators": [100, 300], "estimator__max_depth": [None, 12]}),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
perf, best_params, preds_val, preds_ext, fitted = [], {}, {}, {}, {}

for name, (est, grid) in MODELS.items():
    log("训练 %s（网格 %d 组 × 5 折）..." % (name, int(np.prod([len(v) for v in grid.values()]))))
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    gs.fit(TR[F18], TR["y"])
    fitted[name] = gs.best_estimator_
    best_params[name] = gs.best_params_
    pv = gs.best_estimator_.predict_proba(VA[F18])[:, 1]
    pe = gs.best_estimator_.predict_proba(EX[F18])[:, 1]
    preds_val[name] = pv; preds_ext[name] = pe
    yv, ye = VA["y"].values, EX["y"].values
    rv = evaluate(VA["T_lm"].values, VA["status_lm"].values, yv, pv, "internal")
    re_ = evaluate(EX["T_lm"].values, EX["status_lm"].values, ye, pe, "external")
    (avl, bvl, uvl) = boot_ci(VA["T_lm"].values, VA["status_lm"].values, yv, pv, B=300)
    (aex, bex, uex) = boot_ci(EX["T_lm"].values, EX["status_lm"].values, ye, pe, B=200)
    for r, (a, b, u) in [(rv, (avl, bvl, uvl)), (re_, (aex, bex, uex))]:
        r["model"] = name; r["cv_best_auc"] = gs.best_score_
        r["auc_lo"], r["auc_hi"] = a[0], a[1]
        r["brier_lo"], r["brier_hi"] = b[0], b[1]
        r["uno_lo"], r["uno_hi"] = u[0], u[1]
        r["params"] = json.dumps(gs.best_params_, ensure_ascii=False)
    perf += [rv, re_]
    log("  %s 完成：CV=%.4f | 内部 AUC=%.4f (%.4f-%.4f) | 外部 AUC=%.4f (%.4f-%.4f)"
        % (name, gs.best_score_, rv["auc"], avl[0], avl[1], re_["auc"], aex[0], aex[1]))

P = pd.DataFrame(perf)
cols = ["model", "tag", "n", "events", "cv_best_auc", "auc", "auc_lo", "auc_hi", "pr_auc", "brier",
        "brier_lo", "brier_hi", "uno_c", "uno_lo", "uno_hi", "auc_exd", "brier_exd",
        "auc_t1", "auc_t3", "auc_t6", "oe", "cal_slope", "cal_int", "hl_stat", "hl_df", "hl_P",
        "thr_youden", "sens", "spec", "ppv", "npv", "f1", "kappa", "params"]
P[cols].to_csv(os.path.join(OUT, "n3_model_performance.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(best_params).T.to_csv(os.path.join(OUT, "n3_gridsearch_best.csv"), encoding="utf-8-sig")

pd.DataFrame(preds_val).assign(y=VA["y"].values).to_csv(os.path.join(OUT, "n3_pred_val.csv"), index=False)
pd.DataFrame(preds_ext).assign(y=EX["y"].values).to_csv(os.path.join(OUT, "n3_pred_ext.csv"), index=False)

dl = pd.concat([delong_pairs(VA["y"].values, {k: preds_val[k] for k in preds_val}).assign(tag="internal"),
                delong_pairs(EX["y"].values, {k: preds_ext[k] for k in preds_ext}).assign(tag="external")])
dl.to_csv(os.path.join(OUT, "n3_delong.csv"), index=False, encoding="utf-8-sig")

# 校准十分位 + DCA + 风险四分位（对全部模型）
cal, dca_rows, q_rows = [], [], []
ths = np.round(np.arange(0.05, 0.651, 0.025), 3)
for tag, D, pr, TT, SS in [("internal", VA, preds_val, VA["T_lm"].values, VA["status_lm"].values),
                           ("external", EX, preds_ext, EX["T_lm"].values, EX["status_lm"].values)]:
    y = D["y"].values
    for k, p in pr.items():
        q = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
        for g in np.unique(q.dropna()):
            msk = (q == g).values
            cal.append(dict(tag=tag, model=k, decile=int(g) + 1, n=int(msk.sum()),
                            pred_mean=float(np.mean(p[msk])),
                            obs_cif=SL.aj_cif(TT[msk], SS[msk], SL.HORIZON, 1)))
        for pt, v in zip(ths, SL.dca(TT, SS, p, ths)):
            dca_rows.append(dict(tag=tag, model=k, threshold=float(pt), net_benefit=float(v)))
        qt = SL.logrank_like_oe(TT, SS, p, q=4); qt.insert(0, "model", k); qt.insert(0, "tag", tag)
        q_rows.append(qt)
    for label in ["all", "none"]:
        prev = float(y.mean())
        for pt in ths:
            nb = (prev - (1 - prev) * (pt / (1 - pt))) if label == "all" else 0.0
            dca_rows.append(dict(tag=tag, model=label, threshold=float(pt), net_benefit=float(nb)))
pd.DataFrame(cal).to_csv(os.path.join(OUT, "n3_calibration_deciles.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(dca_rows).to_csv(os.path.join(OUT, "n3_dca.csv"), index=False, encoding="utf-8-sig")
pd.concat(q_rows).to_csv(os.path.join(OUT, "n3_risk_quartiles.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 预设敏感性
best = P[(P["tag"] == "internal")].sort_values("auc", ascending=False).iloc[0]["model"]
log("主模型（内部 AUC 最高）：%s → 敏感性分析" % best)
sens = []
for fname, feats in [("XOI 版（+基线 XOI）", F19), ("λ.min 37 变量集", F37)]:
    est = fitted[best]
    kind = best
    if kind == "LR":
        base = Pipeline([("sc", z()), ("m", LogisticRegression(max_iter=3000, random_state=RS))])
        base.set_params(**{"m__C": best_params["LR"].get("m__C", 1.0)})
    elif kind == "RF":
        base = RandomForestClassifier(random_state=RS, n_jobs=-1, **best_params["RF"])
    elif kind == "XGB":
        base = XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist",
                             **best_params["XGB"])
    elif kind == "LGBM":
        base = LGBMClassifier(random_state=RS, n_jobs=-1, verbose=-1, **best_params["LGBM"])
    elif kind == "SVM":
        base = Pipeline([("sc", z()), ("m", SVC(probability=True, random_state=RS, **best_params["SVM"]))])
    else:
        base = RandomForestClassifier(random_state=RS, n_jobs=-1, n_estimators=300)
    base.fit(TR[feats], TR["y"])
    pv = base.predict_proba(VA[feats])[:, 1]; pe = base.predict_proba(EX[feats])[:, 1]
    for tag, D, p in [("internal", VA, pv), ("external", EX, pe)]:
        y = D["y"].values
        nri, idi = SL.nri_idi(y, p, preds_val[best] if tag == "internal" else preds_ext[best])
        sens.append(dict(sensitivity=fname, tag=tag, model=best, n_feat=len(feats),
                         auc=roc_auc_score(y, p), d_auc=roc_auc_score(y, p) -
                         roc_auc_score(y, preds_val[best] if tag == "internal" else preds_ext[best]),
                         nri=nri, idi=idi, brier=brier_score_loss(y, p)))
pd.DataFrame(sens).to_csv(os.path.join(OUT, "n3_sensitivity.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]; plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (tag, D, pr) in zip(axes, [("内部验证集（MIMIC-IV）", VA, preds_val), ("外部验证集（eICU-CRD）", EX, preds_ext)]):
    y = D["y"].values
    for k, p in pr.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.2, label="%s (AUC %.3f)" % (k, roc_auc_score(y, p)))
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=0.8)
    ax.set_xlabel("1 − 特异度"); ax.set_ylabel("敏感度"); ax.set_title(tag, fontsize=10); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n3_roc.png"), dpi=300)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
cdf = pd.DataFrame(cal)
for ax, tag in zip(axes, ["internal", "external"]):
    for k in preds_val:
        s = cdf[(cdf.tag == tag) & (cdf.model == k)]
        ax.plot(s["pred_mean"], s["obs_cif"], "o-", ms=3, lw=1, label=k)
    ax.plot([0, 0.7], [0, 0.7], "--", c="gray", lw=0.8)
    ax.set_xlabel("预测 7 d AKI 风险"); ax.set_ylabel("观测累积发生率（AJ）")
    ax.set_title("校准曲线 · " + ("内部验证集" if tag == "internal" else "外部验证集"), fontsize=10); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n3_calibration.png"), dpi=300)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
dd = pd.DataFrame(dca_rows)
for ax, tag in zip(axes, ["internal", "external"]):
    for k in list(preds_val) + ["all", "none"]:
        s = dd[(dd.tag == tag) & (dd.model == k)].sort_values("threshold")
        if s.empty: continue
        ls = "--" if k in ("all", "none") else "-"
        ax.plot(s["threshold"], s["net_benefit"], ls, lw=1.1, label=k)
    ax.set_ylim(-0.05, 0.25); ax.set_xlabel("阈值概率"); ax.set_ylabel("净获益")
    ax.set_title("决策曲线 · " + ("内部验证集" if tag == "internal" else "外部验证集"), fontsize=10)
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n3_dca.png"), dpi=300)

qq = pd.concat(q_rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, tag in zip(axes, ["internal", "external"]):
    s = qq[qq.tag == tag]
    w = 0.8 / s["model"].nunique(); x = np.arange(4)
    for i, k in enumerate(sorted(s["model"].unique())):
        ss = s[s.model == k].sort_values("quartile")
        ax.bar(x + i * w, ss["obs_rate"], w, label=k)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylabel("观测 AKI 发生率"); ax.set_title("风险四分位分层 · " + ("内部" if tag == "internal" else "外部"), fontsize=10)
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n3_risk_quartiles.png"), dpi=300)

import joblib
joblib.dump({"model": fitted[best], "features": F18, "name": best}, os.path.join(OUT, "n3_best_model.pkl"))

log("主表：")
log(P[P.tag == "internal"][["model", "auc", "auc_lo", "auc_hi", "pr_auc", "brier", "uno_c", "auc_exd", "oe"]].round(4).to_string(index=False))
log(P[P.tag == "external"][["model", "auc", "auc_lo", "auc_hi", "pr_auc", "brier", "uno_c", "auc_exd", "oe"]].round(4).to_string(index=False))
log("敏感性："); log(pd.DataFrame(sens).round(4).to_string(index=False))
log("全部完成 -> %s" % OUT)
