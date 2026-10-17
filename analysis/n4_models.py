# -*- coding: utf-8 -*-
"""
n4_models.py —— N4：扩展变量集上的 8 模型矩阵 + 外部再校准 + 决策阈值分析
① 8 模型同网格重训（流程与 N3 一致）
② 外部再校准：eICU 50/50 拆为再校准集/评价集；intercept-only、Platt、isotonic 三法
③ 阈值分析：固定敏感度/特异度下的 PPV/NPV；预测风险分层（10%/20%/30%）
产出：n4_model_performance.csv / n4_recalibration.csv / n4_thresholds.csv / n4_*.png / n4_best_model.pkl
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, cohen_kappa_score, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import cn2_survival_lib as SL

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
RS = 123
t0 = time.time()
def log(*a): print("[%6.1f min] " % ((time.time() - t0) / 60) + " ".join(str(x) for x in a), flush=True)

TR = pd.read_csv(os.path.join(OUT, "n4_train.csv"), low_memory=False)
VA = pd.read_csv(os.path.join(OUT, "n4_val.csv"), low_memory=False)
EX = pd.read_csv(os.path.join(OUT, "n4_ext.csv"), low_memory=False)
sel = pd.read_csv(os.path.join(OUT, "n4_selected_variables.csv"))
F = [c for c in sel.loc[sel["primary_intersection"] == True, "var"].tolist() if c in TR.columns]  # noqa: E712
NEW = [c for c in F if c.endswith("_delta") or c in ("cr_ratio", "cr_rise", "bun_cr_delta", "sbp_rel_delta")]
log("N4 最终变量集 %d 个（其中时间趋势/派生 %d 个：%s）" % (len(F), len(NEW), "、".join(NEW)))

def z(): return StandardScaler()
MODELS = {
    "LR": (Pipeline([("sc", z()), ("m", LogisticRegression(max_iter=3000, random_state=RS))]),
           {"m__C": [0.01, 0.1, 1.0, 10.0]}),
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
    "BAG": (BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS), random_state=RS, n_jobs=-1),
            {"n_estimators": [100, 300], "estimator__max_depth": [None, 12]}),
}

def hl_test(y, p, g=10):
    q = pd.qcut(pd.Series(p), g, labels=False, duplicates="drop")
    o1 = pd.Series(y).groupby(q).sum(); e1 = pd.Series(p).groupby(q).sum(); n = pd.Series(y).groupby(q).size()
    stat = float((((o1 - e1) ** 2) / np.maximum(e1, 1e-9) + (((n - o1) - (n - e1)) ** 2) / np.maximum(n - e1, 1e-9)).sum())
    from scipy.stats import chi2
    dfree = max(len(n) - 2, 1)
    return stat, dfree, float(1 - chi2.cdf(stat, dfree))

def calib_slope(y, p):
    import statsmodels.api as sm
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    try:
        f = sm.Logit(y, sm.add_constant(lp)).fit(disp=0); return float(f.params[1]), float(f.params[0])
    except Exception:
        return np.nan, np.nan

def evaluate(T, S, y, p, tag, extra=None):
    m = SL.metrics(T, S, p)
    thr = SL.__dict__ and None
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(y, p); thr = float(th[np.argmax(tpr - fpr)]); yh = (p >= thr).astype(int)
    cm = confusion_matrix(y, yh); tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
    hl, hdf, hlp = hl_test(y, p); cs, ci = calib_slope(y, p)
    d = dict(tag=tag, n=len(y), events=int(y.sum()), auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
             brier=brier_score_loss(y, p), uno_c=m["C_uno"], auc_exd=m["AUC_exd"], brier_exd=m["Brier_exd"],
             auc_t1=m["AUC_t1"], auc_t3=m["AUC_t3"], auc_t6=m["AUC_t6"], oe=m["O_E_7d"],
             cal_slope=cs, cal_int=ci, hl_stat=hl, hl_df=hdf, hl_P=hlp, thr_youden=thr,
             sens=tp / (tp + fn) if (tp + fn) else np.nan, spec=tn / (tn + fp) if (tn + fp) else np.nan,
             ppv=tp / (tp + fp) if (tp + fp) else np.nan, npv=tn / (tn + fn) if (tn + fn) else np.nan,
             f1=f1_score(y, yh), kappa=cohen_kappa_score(y, yh))
    if extra: d.update(extra)
    return d

def boot_ci(T, S, y, p, B=300, seed=20261016):
    rng = np.random.default_rng(seed); a, b, u = [], [], []
    for _ in range(B):
        bi = rng.integers(0, len(y), len(y))
        if y[bi].sum() == 0 or y[bi].sum() == len(bi): continue
        a.append(roc_auc_score(y[bi], p[bi])); b.append(brier_score_loss(y[bi], p[bi]))
        try: u.append(SL.metrics(T[bi], S[bi], p[bi], with_binary=False)["C_uno"])
        except Exception: pass
    f = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if len(v) else (np.nan, np.nan)
    return f(a), f(b), f(u)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
perf, pv_all, pe_all, fitted, bparams = [], {}, {}, {}, {}
for name, (est, grid) in MODELS.items():
    log("训练 %s ..." % name)
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    gs.fit(TR[F], TR["y"]); fitted[name] = gs.best_estimator_; bparams[name] = gs.best_params_
    pv = gs.best_estimator_.predict_proba(VA[F])[:, 1]; pe = gs.best_estimator_.predict_proba(EX[F])[:, 1]
    pv_all[name] = pv; pe_all[name] = pe
    rv = evaluate(VA["T_lm"].values, VA["status_lm"].values, VA["y"].values, pv, "internal",
                  {"model": name, "cv_best_auc": gs.best_score_, "params": json.dumps(gs.best_params_, ensure_ascii=False)})
    re_ = evaluate(EX["T_lm"].values, EX["status_lm"].values, EX["y"].values, pe, "external",
                   {"model": name, "cv_best_auc": gs.best_score_, "params": json.dumps(gs.best_params_, ensure_ascii=False)})
    (al, bl, ul) = boot_ci(VA["T_lm"].values, VA["status_lm"].values, VA["y"].values, pv, 300)
    (ae, be, ue) = boot_ci(EX["T_lm"].values, EX["status_lm"].values, EX["y"].values, pe, 200)
    for r, (a, b, u) in [(rv, (al, bl, ul)), (re_, (ae, be, ue))]:
        r["auc_lo"], r["auc_hi"] = a; r["brier_lo"], r["brier_hi"] = b; r["uno_lo"], r["uno_hi"] = u
    perf += [rv, re_]
    log("  %s：CV %.4f | 内部 AUC %.4f (%.4f-%.4f) | 外部 AUC %.4f (%.4f-%.4f)"
        % (name, gs.best_score_, rv["auc"], al[0], al[1], re_["auc"], ae[0], ae[1]))

P = pd.DataFrame(perf)
P.to_csv(os.path.join(OUT, "n4_model_performance.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(pv_all).assign(y=VA["y"].values).to_csv(os.path.join(OUT, "n4_pred_val.csv"), index=False)
pd.DataFrame(pe_all).assign(y=EX["y"].values).to_csv(os.path.join(OUT, "n4_pred_ext.csv"), index=False)
pd.DataFrame(bparams).T.to_csv(os.path.join(OUT, "n4_gridsearch_best.csv"), encoding="utf-8-sig")

# ---------------- 外部再校准（eICU 50/50 拆分） ----------------
rng = np.random.default_rng(RS)
idx = np.arange(len(EX)); rec_i, ev_i = [], []
for v in [0, 1]:
    sub = idx[EX["y"].values == v]; rng.shuffle(sub)
    half = len(sub) // 2; rec_i += list(sub[:half]); ev_i += list(sub[half:])
rec_i = np.array(sorted(rec_i)); ev_i = np.array(sorted(ev_i))
log("再校准集 n=%d（AKI %d）／评价集 n=%d（AKI %d）"
    % (len(rec_i), EX["y"].values[rec_i].sum(), len(ev_i), EX["y"].values[ev_i].sum()))
rows_recal = []
for name, pe in pe_all.items():
    p_rec, p_ev = pe[rec_i], pe[ev_i]; y_rec, y_ev = EX["y"].values[rec_i], EX["y"].values[ev_i]
    lr = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    # 原始
    for tag, p in [("原始", p_ev)]:
        cs, ci = calib_slope(y_ev, p); oe = SL.aj_cif(EX["T_lm"].values[ev_i], EX["status_lm"].values[ev_i], SL.HORIZON, 1) / max(p.mean(), 1e-9)
        rows_recal.append(dict(model=name, method=tag, brier=brier_score_loss(y_ev, p), oe=oe, cal_slope=cs, cal_int=ci,
                               auc=roc_auc_score(y_ev, p)))
    # intercept-only：p' = expit(logit(p) + b)
    import statsmodels.api as sm
    try:
        fit = sm.Logit(y_rec, sm.add_constant(lr(p_rec), has_constant="add")).fit(disp=0)
        b = float(fit.params[0]); p2 = 1 / (1 + np.exp(-(lr(p_ev) + b)))
        cs, ci = calib_slope(y_ev, p2)
        oe = SL.aj_cif(EX["T_lm"].values[ev_i], EX["status_lm"].values[ev_i], SL.HORIZON, 1) / max(p2.mean(), 1e-9)
        rows_recal.append(dict(model=name, method="截距再校准", brier=brier_score_loss(y_ev, p2), oe=oe, cal_slope=cs, cal_int=ci,
                               auc=roc_auc_score(y_ev, p2)))
    except Exception as e:
        log("  截距再校准失败 %s: %s" % (name, e))
    # Platt：p' = expit(a*logit(p)+b)
    try:
        fit = sm.Logit(y_rec, sm.add_constant(lr(p_rec), has_constant="add")).fit(disp=0)
        a_, b_ = float(fit.params[1]), float(fit.params[0]); p3 = 1 / (1 + np.exp(-(a_ * lr(p_ev) + b_)))
        cs, ci = calib_slope(y_ev, p3)
        oe = SL.aj_cif(EX["T_lm"].values[ev_i], EX["status_lm"].values[ev_i], SL.HORIZON, 1) / max(p3.mean(), 1e-9)
        rows_recal.append(dict(model=name, method="Platt 再校准", brier=brier_score_loss(y_ev, p3), oe=oe, cal_slope=cs, cal_int=ci,
                               auc=roc_auc_score(y_ev, p3)))
    except Exception as e:
        log("  Platt 失败 %s: %s" % (name, e))
    # isotonic
    try:
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_rec, y_rec)
        p4 = iso.predict(p_ev); p4 = np.clip(p4, 1e-6, 1 - 1e-6)
        cs, ci = calib_slope(y_ev, p4)
        oe = SL.aj_cif(EX["T_lm"].values[ev_i], EX["status_lm"].values[ev_i], SL.HORIZON, 1) / max(p4.mean(), 1e-9)
        rows_recal.append(dict(model=name, method="Isotonic 再校准", brier=brier_score_loss(y_ev, p4), oe=oe,
                               cal_slope=cs, cal_int=ci, auc=roc_auc_score(y_ev, p4)))
    except Exception as e:
        log("  Isotonic 失败 %s: %s" % (name, e))
RC = pd.DataFrame(rows_recal)
RC.to_csv(os.path.join(OUT, "n4_recalibration.csv"), index=False, encoding="utf-8-sig")

# ---------------- 决策阈值分析（主模型） ----------------
best = P[P.tag == "internal"].sort_values("auc", ascending=False).iloc[0]["model"]
log("主模型：%s" % best)
thr_rows = []
for tag, D, p in [("internal", VA, pv_all[best]), ("external", EX, pe_all[best])]:
    y = D["y"].values
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(y, p)
    for target in [0.90, 0.85, 0.80]:
        k = np.argmin(np.abs(tpr - target)); t = float(th[k]); yh = (p >= t).astype(int)
        cm = confusion_matrix(y, yh); tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
        thr_rows.append(dict(tag=tag, rule="敏感度≈%.0f%%" % (target * 100), threshold=t, sens=tp / (tp + fn),
                             spec=tn / (tn + fp), ppv=tp / (tp + fp) if (tp + fp) else np.nan,
                             npv=tn / (tn + fn) if (tn + fn) else np.nan, flagged_pct=100 * yh.mean()))
    for target in [0.90, 0.80, 0.70]:
        k = np.argmin(np.abs((1 - fpr) - target)); t = float(th[k]); yh = (p >= t).astype(int)
        cm = confusion_matrix(y, yh); tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
        thr_rows.append(dict(tag=tag, rule="特异度≈%.0f%%" % (target * 100), threshold=t, sens=tp / (tp + fn),
                             spec=tn / (tn + fp), ppv=tp / (tp + fp) if (tp + fp) else np.nan,
                             npv=tn / (tn + fn) if (tn + fn) else np.nan, flagged_pct=100 * yh.mean()))
    for c in [0.10, 0.20, 0.30]:
        yh = (p >= c).astype(int)
        cm = confusion_matrix(y, yh); tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan,) * 4
        thr_rows.append(dict(tag=tag, rule="固定阈值 ≥%.0f%%" % (c * 100), threshold=c, sens=tp / (tp + fn),
                             spec=tn / (tn + fp), ppv=tp / (tp + fp) if (tp + fp) else np.nan,
                             npv=tn / (tn + fn) if (tn + fn) else np.nan, flagged_pct=100 * yh.mean()))
TH = pd.DataFrame(thr_rows); TH.to_csv(os.path.join(OUT, "n4_thresholds.csv"), index=False, encoding="utf-8-sig")

# ---------------- 图：再校准前后 + 阈值曲线 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]; plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
y_ev = EX["y"].values[ev_i]
q = pd.qcut(pd.Series(pe_all[best][ev_i]), 10, labels=False, duplicates="drop")
obs = [SL.aj_cif(EX["T_lm"].values[ev_i][(q == g).values], EX["status_lm"].values[ev_i][(q == g).values],
                 SL.HORIZON, 1) for g in np.unique(q.dropna())]
pm = [float(np.mean(pe_all[best][ev_i][(q == g).values])) for g in np.unique(q.dropna())]
axes[0].plot([0, 0.7], [0, 0.7], "--", c="gray", lw=0.8)
axes[0].plot(pm, obs, "o-", c="#c0392b", ms=4, lw=1.2, label="原始")
lr_ = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
import statsmodels.api as sm
fit = sm.Logit(EX["y"].values[rec_i], sm.add_constant(lr_(pe_all[best][rec_i]), has_constant="add")).fit(disp=0)
p2 = 1 / (1 + np.exp(-(lr_(pe_all[best][ev_i]) + float(fit.params[0]))))
q2 = pd.qcut(pd.Series(p2), 10, labels=False, duplicates="drop")
obs2 = [SL.aj_cif(EX["T_lm"].values[ev_i][(q2 == g).values], EX["status_lm"].values[ev_i][(q2 == g).values],
                  SL.HORIZON, 1) for g in np.unique(q2.dropna())]
pm2 = [float(np.mean(p2[(q2 == g).values])) for g in np.unique(q2.dropna())]
axes[0].plot(pm2, obs2, "s-", c="#2980b9", ms=4, lw=1.2, label="截距再校准后")
axes[0].set_xlabel("预测 7 d AKI 风险"); axes[0].set_ylabel("观测累积发生率（AJ）")
axes[0].set_title("外部集再校准前后（%s，评价集 n=%d）" % (best, len(ev_i)), fontsize=10); axes[0].legend(fontsize=8)
s = TH[(TH.tag == "external") & (TH.rule.str.startswith("敏感度"))].sort_values("threshold")
axes[1].plot(s["threshold"], s["ppv"], "o-", label="PPV")
axes[1].plot(s["threshold"], s["npv"], "s-", label="NPV")
axes[1].set_xlabel("阈值"); axes[1].set_ylabel("概率"); axes[1].set_title("外部集固定敏感度下的 PPV/NPV（%s）" % best, fontsize=10)
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n4_recalibration_thresholds.png"), dpi=300)

import joblib
joblib.dump({"model": fitted[best], "features": F, "name": best}, os.path.join(OUT, "n4_best_model.pkl"))

log("主表（内部）："); log(P[P.tag == "internal"][["model", "auc", "auc_lo", "auc_hi", "brier", "uno_c", "auc_exd", "oe"]].round(4).to_string(index=False))
log("主表（外部）："); log(P[P.tag == "external"][["model", "auc", "auc_lo", "auc_hi", "brier", "uno_c", "auc_exd", "oe"]].round(4).to_string(index=False))
log("再校准（RF/XGB/LGBM）："); log(RC[RC.model.isin(["RF", "XGB", "LGBM"])].round(4).to_string(index=False))
log("阈值分析："); log(TH.round(3).to_string(index=False))
log("完成 -> %s" % OUT)
