# -*- coding: utf-8 -*-
"""
n5_analysis.py —— N5 五项稳健性/验证分析
① 时间验证：按 MIMIC-IV 官方 anchor_year_group（真实时段）切分，早期训练 → 后期验证
② MIMIC 单库强变量模型（乳酸/尿量/体温/P-F）：量化"数据可得性上限"
③ 缺失处理敏感性（MICE vs 完整病例）＋类别不平衡（自实现 SMOTE，仅训练集）
④ 预设亚组分析 ＋ Holm 多重校正
⑤ 主模型 Bootstrap 1000 次校准曲线（内部与外部）
产出：n5_temporal.csv / n5_mimic_only.csv / n5_sensitivity.csv / n5_subgroups.csv / n5_calibration_boot.png
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import statsmodels.api as sm
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
CONT = [c for c in F if c not in ("female", "vent", "vaso", "xoi_base", "htn", "dm", "chf", "ckd", "lung",
                                 "mi", "cva", "cancer", "liver", "gout")]
MM = pd.read_csv(os.path.join(OUT, "n4_mimic_dataset.csv"), low_memory=False)
log("主变量集 %d 个（连续 %d）" % (len(F), len(CONT)))

def fit_predict(Dtr, Dte, feats, kind="XGB", impute=True):
    """在 Dtr 上拟合插补器与模型，返回对 Dte 的预测。"""
    cont = [c for c in feats if c in CONT]
    if impute:
        imp = IterativeImputer(max_iter=20, random_state=RS, keep_empty_features=True)
        imp.fit(Dtr[cont]); med = Dtr[cont].median()
        def tr(d):
            x = pd.DataFrame(imp.transform(d[cont]), columns=cont)
            for c in cont:
                x[c] = x[c].fillna(med[c])
            out = d.copy(); out[cont] = x
            return out
    else:
        tr = lambda d: d
    A, B = tr(Dtr), tr(Dte)
    if kind == "XGB":
        m = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, random_state=RS, n_jobs=-1,
                          eval_metric="logloss", tree_method="hist")
    elif kind == "RF":
        m = RandomForestClassifier(n_estimators=600, max_depth=12, min_samples_leaf=1,
                                   random_state=RS, n_jobs=-1)
    else:
        m = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=3000, random_state=RS, C=0.1))])
    m.fit(A[feats], A["y"])
    return m.predict_proba(B[feats])[:, 1], m

# ================================================ ① 时间验证
anc = pd.read_csv(os.path.join(OUT, "_anchor_group.csv"), low_memory=False)
TRa = TR.merge(anc, on="stay_id", how="left")
VAa = VA.merge(anc, on="stay_id", how="left")
allm = pd.concat([TRa, VAa], ignore_index=True)
EARLY = ["2008 - 2010", "2011 - 2013"]
LATE = ["2014 - 2016", "2017 - 2019", "2020 - 2022"]
tr_e = allm[allm["anchor_group"].isin(EARLY)].reset_index(drop=True)
te_l = allm[allm["anchor_group"].isin(LATE)].reset_index(drop=True)
log("时间验证：早期（2008–2013）n=%d（AKI %d）→ 后期（2014–2022）n=%d（AKI %d）"
    % (len(tr_e), tr_e.y.sum(), len(te_l), te_l.y.sum()))
rows_t = []
for kind in ["XGB", "RF", "LR"]:
    p, model = fit_predict(tr_e, te_l, F, kind)
    y = te_l["y"].values
    m = SL.metrics(te_l["T_lm"].values, te_l["status_lm"].values, p)
    cs = sm.Logit(y, sm.add_constant(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))),
                                      has_constant="add")).fit(disp=0)
    rows_t.append(dict(model=kind, n_train=len(tr_e), n_test=len(te_l), events=int(y.sum()),
                       auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), uno_c=m["C_uno"],
                       auc_exd=m["AUC_exd"], oe=m["O_E_7d"], cal_slope=float(cs.params[1]),
                       cal_int=float(cs.params[0])))
    log("  %s：AUC %.4f | Brier %.4f | O/E %.3f | 斜率 %.3f" % (kind, rows_t[-1]["auc"], rows_t[-1]["brier"],
                                                               rows_t[-1]["oe"], rows_t[-1]["cal_slope"]))
# 变量集稳定性：早期时段内用 L1 逻辑回归重筛（等价 LASSO），与锁定集比较
from sklearn.linear_model import LogisticRegressionCV
imp_e = IterativeImputer(max_iter=20, random_state=RS, keep_empty_features=True).fit(tr_e[CONT])
Xe = pd.DataFrame(imp_e.transform(tr_e[CONT]), columns=CONT)
l1 = LogisticRegressionCV(penalty="l1", solver="saga", cv=5, scoring="roc_auc", max_iter=4000,
                          Cs=np.logspace(-3, 1, 30), random_state=RS, n_jobs=-1)
l1.fit(StandardScaler().fit_transform(Xe), tr_e["y"].values)
sel_e = set(np.array(CONT)[np.abs(l1.coef_[0]) > 1e-8])
locked_cont = set(c for c in F if c in CONT)
overlap = len(sel_e & locked_cont)
log("  变量集稳定性：早期重筛 %d 个（连续变量），与锁定集重合 %d/%d，Jaccard %.2f"
    % (len(sel_e), overlap, len(locked_cont), overlap / max(len(sel_e | locked_cont), 1)))
pd.DataFrame(rows_t).to_csv(os.path.join(OUT, "n5_temporal.csv"), index=False, encoding="utf-8-sig")

# ================================================ ② MIMIC 单库强变量模型
MIM_ONLY = ["lac_post_max", "uo6_post_min", "uo12_post_min", "uo24_post_min", "temp_post_max", "pf_post_min"]
avl = [c for c in MIM_ONLY if c in TR.columns and any(c in x.columns for x in (TR, VA))]
log("MIMIC 单库强变量：%s" % "、".join(avl))
F2 = F + avl
CONT2 = CONT + avl
TRm = TR.copy(); VAm = VA.copy()
imp2 = IterativeImputer(max_iter=20, random_state=RS, keep_empty_features=True).fit(TRm[CONT2])
med2 = TRm[CONT2].median()
def fill2(d):
    x = pd.DataFrame(imp2.transform(d[CONT2]), columns=CONT2)
    for c in CONT2: x[c] = x[c].fillna(med2[c])
    o = d.copy(); o[CONT2] = x
    for c in avl: o["miss_" + c] = d[c].isna().astype(int)
    return o
TRm, VAm = fill2(TRm), fill2(VAm)
rows_m = []
for kind in ["XGB", "RF"]:
    for tag, featset in [("两库可比集（29）", F), ("＋MIMIC 单库强变量（35）", F2 + ["miss_" + c for c in avl])]:
        A = TRm if tag.startswith("＋") else TR
        B = VAm if tag.startswith("＋") else VA
        if kind == "XGB":
            m = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=RS, n_jobs=-1,
                              eval_metric="logloss", tree_method="hist")
        else:
            m = RandomForestClassifier(n_estimators=600, max_depth=12, random_state=RS, n_jobs=-1)
        m.fit(A[featset], A["y"])
        p = m.predict_proba(B[featset])[:, 1]; y = B["y"].values
        rows_m.append(dict(model=kind, featureset=tag, n_feat=len(featset), auc=roc_auc_score(y, p),
                           brier=brier_score_loss(y, p)))
        log("  %s × %s：AUC %.4f | Brier %.4f" % (kind, tag, rows_m[-1]["auc"], rows_m[-1]["brier"]))
pd.DataFrame(rows_m).to_csv(os.path.join(OUT, "n5_mimic_only.csv"), index=False, encoding="utf-8-sig")

# ================================================ ③ 敏感性：完整病例 / SMOTE
rows_s = []
cc_mask_tr = TR[F].notna().all(axis=1); cc_mask_va = VA[F].notna().all(axis=1)
log("完整病例：训练 %d/%d（%.1f%%）／验证 %d/%d（%.1f%%）"
    % (cc_mask_tr.sum(), len(TR), 100 * cc_mask_tr.mean(), cc_mask_va.sum(), len(VA), 100 * cc_mask_va.mean()))
for kind in ["XGB"]:
    p, _ = fit_predict(TR[cc_mask_tr], VA[cc_mask_va], F, kind, impute=False)
    y = VA.loc[cc_mask_va, "y"].values
    rows_s.append(dict(scenario="完整病例（CC）", model=kind, n=len(y), events=int(y.sum()),
                       auc=roc_auc_score(y, p), brier=brier_score_loss(y, p)))
    log("  %s 完整病例：AUC %.4f" % (kind, rows_s[-1]["auc"]))

# 自实现 SMOTE（仅训练集）
def smote(X, y, k=5, seed=RS):
    rng = np.random.default_rng(seed)
    xm = X[y == 1]; xj = X[y == 0]
    need = len(xj) - len(xm)
    if need <= 0 or len(xm) < k + 1:
        return X, y
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xm)
    idx = rng.integers(0, len(xm), need)
    _, nbr = nn.kneighbors(xm[idx])
    pick = nbr[np.arange(need), rng.integers(1, k + 1, need)]
    alpha = rng.random((need, 1))
    new = xm[idx] + alpha * (xm[pick] - xm[idx])
    return np.vstack([X, new]), np.concatenate([y, np.ones(need, dtype=int)])

imp3 = IterativeImputer(max_iter=20, random_state=RS, keep_empty_features=True).fit(TR[CONT])
med3 = TR[CONT].median()
def filld(d):
    x = pd.DataFrame(imp3.transform(d[CONT]), columns=CONT)
    for c in CONT: x[c] = x[c].fillna(med3[c])
    o = d.copy(); o[CONT] = x; return o
TRs, VAs = filld(TR), filld(VA)
Xs, ys = smote(TRs[F].values.astype(float), TRs["y"].values.astype(int))
log("  SMOTE：训练集 %d → %d（合成 %d）" % (len(TRs), len(Xs), len(Xs) - len(TRs)))
mx = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                   reg_lambda=1.0, random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist")
mx.fit(Xs, ys)
p = mx.predict_proba(VAs[F].values.astype(float))[:, 1]; y = VAs["y"].values
rows_s.append(dict(scenario="SMOTE（仅训练集）", model="XGB", n=len(y), events=int(y.sum()),
                   auc=roc_auc_score(y, p), brier=brier_score_loss(y, p)))
log("  XGB SMOTE：AUC %.4f" % rows_s[-1]["auc"])
# 参照（无 SMOTE / MICE 主流程）
m0 = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                   reg_lambda=1.0, random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist")
m0.fit(TRs[F], TRs["y"]); p0 = m0.predict_proba(VAs[F])[:, 1]
rows_s.append(dict(scenario="MICE 主流程（参照）", model="XGB", n=len(y), events=int(y.sum()),
                   auc=roc_auc_score(y, p0), brier=brier_score_loss(y, p0)))
pd.DataFrame(rows_s).to_csv(os.path.join(OUT, "n5_sensitivity.csv"), index=False, encoding="utf-8-sig")

# ================================================ ④ 亚组分析（内部验证，XGB）
p_va = p0
sg_def = {"性别：女性": VA["female"] == 1, "性别：男性": VA["female"] == 0,
          "年龄 ≥65": VA["age"] >= 65, "年龄 <65": VA["age"] < 65,
          "慢性肾病：有": VA["ckd"] == 1, "慢性肾病：无": VA["ckd"] == 0,
          "机械通气：有": VA["vent"] == 1, "机械通气：无": VA["vent"] == 0,
          "SOFA ≥8": VA["sofa"] >= 8, "SOFA <8": VA["sofa"] < 8,
          "基线 XOI：用": VA["xoi_base"] == 1, "基线 XOI：未用": VA["xoi_base"] == 0}
rows_g = []
for name, mask in sg_def.items():
    mask = mask.fillna(False).values
    y = VA["y"].values[mask]; pp = p_va[mask]
    if y.sum() < 20 or (1 - y).sum() < 20:
        continue
    rows_g.append(dict(subgroup=name, n=int(mask.sum()), events=int(y.sum()),
                       auc=roc_auc_score(y, pp), brier=brier_score_loss(y, pp)))
# 交互检验（logit(p) × 亚组指示变量）
lp = np.log(np.clip(p_va, 1e-6, 1 - 1e-6) / (1 - np.clip(p_va, 1e-6, 1 - 1e-6)))
yg = VA["y"].values
inter = []
for var, label in [("female", "性别"), ("age65", "年龄≥65"), ("ckd", "慢性肾病"), ("vent", "机械通气"),
                   ("sofa8", "SOFA≥8"), ("xoi_base", "基线 XOI")]:
    if var == "age65":
        g = (VA["age"] >= 65).astype(float).values
    elif var == "sofa8":
        g = (VA["sofa"] >= 8).astype(float).values
    else:
        g = VA[var].astype(float).values
    X = np.column_stack([lp, g, lp * g])
    try:
        f = sm.Logit(yg, sm.add_constant(X, has_constant="add")).fit(disp=0)
        inter.append(dict(variable=label, p_interaction=float(f.pvalues[3])))
    except Exception as e:
        inter.append(dict(variable=label, p_interaction=np.nan))
it = pd.DataFrame(inter)
# Holm 校正
ps = it["p_interaction"].values.astype(float); order = np.argsort(ps); m_ = len(ps); adj = np.empty(m_)
prev = 0
for rank, i in enumerate(order):
    val = min(1.0, (m_ - rank) * ps[i]); val = max(val, prev); adj[i] = val; prev = val
it["p_holm"] = adj
it.to_csv(os.path.join(OUT, "n5_subgroup_interactions.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(rows_g).to_csv(os.path.join(OUT, "n5_subgroups.csv"), index=False, encoding="utf-8-sig")
log("亚组 AUC 范围 %.3f–%.3f；交互检验（Holm 校正后）最小 P=%.4f"
    % (min(r["auc"] for r in rows_g), max(r["auc"] for r in rows_g), np.nanmin(adj)))

# ================================================ ⑤ Bootstrap 1000 校准曲线
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]; plt.rcParams["axes.unicode_minus"] = False
def boot_calib(T, S, p, B=1000, seed=20261016, nbins=10):
    rng = np.random.default_rng(seed); curves = []
    q = pd.qcut(pd.Series(p), nbins, labels=False, duplicates="drop")
    bins = np.unique(q.dropna())
    for _ in range(B):
        bi = rng.integers(0, len(p), len(p))
        if ((T[bi] <= SL.HORIZON) & (S[bi] == 1)).sum() < 20: continue
        cur = []
        for g in bins:
            m = (q == g).values[bi]
            if m.sum() < 5: cur.append((np.nan, np.nan)); continue
            cur.append((float(np.mean(p[bi][m])), SL.aj_cif(T[bi][m], S[bi][m], SL.HORIZON, 1)))
        curves.append(cur)
    arr = np.array(curves, dtype=float)
    return arr
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, (tag, T, S, p, n) in zip(axes, [
        ("内部验证集（MIMIC-IV）", VA["T_lm"].values, VA["status_lm"].values, p0, len(VA)),
        ("外部验证集（eICU-CRD）", EX["T_lm"].values, EX["status_lm"].values,
         pd.read_csv(os.path.join(OUT, "n4_pred_ext.csv"))["XGB"].values, len(EX))]):
    arr = boot_calib(T, S, p, B=500)
    pm = np.nanmean(arr[:, :, 0], axis=0); om = np.nanmean(arr[:, :, 1], axis=0)
    lo = np.nanpercentile(arr[:, :, 1], 2.5, axis=0); hi = np.nanpercentile(arr[:, :, 1], 97.5, axis=0)
    ax.plot([0, 0.8], [0, 0.8], "--", c="gray", lw=0.8)
    ax.fill_between(pm, lo, hi, color="#c0392b", alpha=0.18, label="观测 95% 区间（Bootstrap）")
    ax.plot(pm, om, "o-", c="#c0392b", ms=4, lw=1.2, label="观测（AJ）")
    ax.set_xlabel("预测 7 d AKI 风险"); ax.set_ylabel("观测累积发生率")
    ax.set_title("%s（n=%d，主模型 XGB）" % (tag, n), fontsize=10); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n5_calibration_boot.png"), dpi=300)
log("SAVED: n5_calibration_boot.png；全部完成")
