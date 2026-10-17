# -*- coding: utf-8 -*-
"""
n6_shap.py —— N6-①：主模型（XGB）SHAP 全套可解释性分析
产出：n6_shap_beeswarm.png、n6_shap_bar.png、n6_shap_dependence.png、n6_shap_waterfall.png、
      n6_shap_importance.csv
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
import shap, joblib

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
LAB = {"agap_delta": "阴离子间隙变化", "agap_post_max": "阴离子间隙最高", "age": "年龄",
       "bun_cr_ratio": "BUN/Cr 比", "chf": "心力衰竭", "ckd": "慢性肾病", "cl_post_min": "氯最低",
       "cr_base": "基线肌酐", "cr_delta": "肌酐变化幅度", "cr_ratio": "肌酐峰值/基线比",
       "cva": "脑卒中", "dm": "糖尿病", "hb_delta": "Hgb 变化幅度", "hb_post_min": "Hgb 最低",
       "hct_delta": "Hct 变化幅度", "hr_delta": "心率变化幅度", "hr_post_max": "心率最高",
       "k_post_max": "钾最高", "liver": "肝病", "mchc_post_max": "MCHC 最高",
       "mcv_post_max": "MCV 最高", "plt_delta": "PLT 变化幅度", "rdw_post_max": "RDW 最高",
       "rr_post_max": "呼吸频率最高", "sbp_delta": "收缩压变化幅度", "sbp_rel_delta": "收缩压相对波动",
       "sofa": "SOFA 评分", "vent": "机械通气", "wbc_post_max": "WBC 最高"}
UNIT = {"cr_base": "mg/dL", "cr_delta": "mg/dL", "cr_ratio": "倍", "bun_cr_ratio": "比",
        "age": "岁", "sofa": "分"}

TR = pd.read_csv(os.path.join(OUT, "n4_train.csv"), low_memory=False)
VA = pd.read_csv(os.path.join(OUT, "n4_val.csv"), low_memory=False)
EX = pd.read_csv(os.path.join(OUT, "n4_ext.csv"), low_memory=False)
bundle = joblib.load(os.path.join(OUT, "n4_best_model.pkl"))
model, F, name = bundle["model"], bundle["features"], bundle["name"]
print("主模型：%s，变量 %d 个" % (name, len(F)))

exp = shap.TreeExplainer(model)
def shap_of(D, n=None, seed=7):
    d = D[F]
    if n and len(d) > n:
        d = d.sample(n=n, random_state=seed)
    return exp.shap_values(d), d

sv_va, X_va = shap_of(VA, n=6000)
sv_ex, X_ex = shap_of(EX, n=5000)
print("SHAP 计算完成：内部 %s，外部 %s" % (sv_va.shape, sv_ex.shape))

imp = pd.DataFrame({
    "var": F, "label": [LAB.get(c, c) for c in F],
    "mean_abs_shap_internal": np.abs(sv_va).mean(axis=0),
    "mean_abs_shap_external": np.abs(sv_ex).mean(axis=0)})
imp["rank_internal"] = imp["mean_abs_shap_internal"].rank(ascending=False).astype(int)
imp["rank_external"] = imp["mean_abs_shap_external"].rank(ascending=False).astype(int)
imp = imp.sort_values("mean_abs_shap_internal", ascending=False)
imp.to_csv(os.path.join(OUT, "n6_shap_importance.csv"), index=False, encoding="utf-8-sig")
print(imp.head(10).round(4).to_string(index=False))

# 1) 蜂群图（内部）
Xl = X_va.rename(columns={c: LAB.get(c, c) for c in F})
plt.figure(figsize=(8.2, 7))
shap.summary_plot(sv_va, Xl, show=False, max_display=18, plot_size=None)
plt.title("SHAP 蜂群图 · 内部验证集（MIMIC-IV）", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n6_shap_beeswarm.png"), dpi=300, bbox_inches="tight"); plt.close()

# 2) 重要性条形图（内外对比）
fig, ax = plt.subplots(figsize=(8.4, 6.6))
top = imp.head(18).iloc[::-1]
y = np.arange(len(top))
ax.barh(y + 0.2, top["mean_abs_shap_internal"], height=0.4, color="#c0392b", label="内部验证集")
ax.barh(y - 0.2, top["mean_abs_shap_external"], height=0.4, color="#2980b9", label="外部验证集")
ax.set_yticks(y); ax.set_yticklabels(top["label"], fontsize=8.5)
ax.set_xlabel("平均 |SHAP| 值"); ax.legend(fontsize=8)
ax.set_title("SHAP 特征重要性（主模型 XGB，29 变量）", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n6_shap_bar.png"), dpi=300); plt.close()

# 3) 依赖图（前 4 变量）
fig, axes = plt.subplots(2, 2, figsize=(10.4, 8))
for ax, v in zip(axes.ravel(), imp["var"].head(4)):
    j = F.index(v)
    x = X_va[v].values; y_ = sv_va[:, j]
    ax.scatter(x, y_, s=4, alpha=0.35, c="#7f8c8d", edgecolors="none")
    order = np.argsort(x); k = max(len(x) // 30, 5)
    xs = pd.Series(x[order]).rolling(k, center=True, min_periods=1).mean()
    ax.plot(xs, pd.Series(y_[order]).rolling(k, center=True, min_periods=1).mean(), c="#c0392b", lw=1.6)
    ax.axhline(0, ls="--", c="gray", lw=0.7)
    ax.set_xlabel("%s%s" % (LAB.get(v, v), ("（%s）" % UNIT[v]) if v in UNIT else ""), fontsize=9)
    ax.set_ylabel("SHAP 值", fontsize=9)
fig.suptitle("SHAP 依赖图（前 4 位变量，内部验证集）", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n6_shap_dependence.png"), dpi=300); plt.close()

# 4) 单例力图（最高危 + 最低危）
p_va = model.predict_proba(VA[F])[:, 1]
hi = int(np.where(VA["y"].values == 1)[0][np.argmax(p_va[VA["y"].values == 1])])
lo = int(np.where(VA["y"].values == 0)[0][np.argmin(p_va[VA["y"].values == 0])])
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, idx, ttl in [(axes[0], hi, "高危急例（实际发生 7 d AKI，预测 %.1f%%）" % (100 * p_va[hi])),
                     (axes[1], lo, "低危急例（未发生 AKI，预测 %.1f%%）" % (100 * p_va[lo]))]:
    sv = exp.shap_values(VA[F].iloc[[idx]])[0]
    order = np.argsort(np.abs(sv))[::-1][:10][::-1]
    ax.barh(np.arange(len(order)), sv[order],
            color=["#c0392b" if s > 0 else "#2980b9" for s in sv[order]])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([LAB.get(F[j], F[j]) for j in order], fontsize=8.5)
    ax.axvline(0, c="gray", lw=0.8)
    ax.set_xlabel("SHAP 值（正=推高风险）", fontsize=9)
    ax.set_title(ttl, fontsize=10)
fig.suptitle("单例力图（前 10 位贡献变量，内部验证集）", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "n6_shap_waterfall.png"), dpi=300); plt.close()

print("\n前 10 位变量（内部/外部平均 |SHAP|）：")
print(imp.head(10)[["label", "mean_abs_shap_internal", "mean_abs_shap_external", "rank_internal", "rank_external"]].round(4).to_string(index=False))
print("\nSAVED: n6_shap_beeswarm.png / n6_shap_bar.png / n6_shap_dependence.png / n6_shap_waterfall.png / n6_shap_importance.csv")
