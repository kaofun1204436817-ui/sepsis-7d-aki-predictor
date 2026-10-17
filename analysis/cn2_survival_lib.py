# -*- coding: utf-8 -*-
"""
cn2_survival_lib.py —— CN2 新稿生存/竞争风险评价库（自实现，无外部生存包）
提供：队列重建、IPCW 删失分布、Aalen–Johansen、Uno's C（BIT 加速）、
时间依赖 AUC、IPCW Brier、NRI/IDI、DCA、校准十分位。
口径：随访 7 d（10 080 min）；cause1=AKI，cause2=死亡（竞争），cause0=出院/行政截断。
"""
import os
import numpy as np
import pandas as pd

HORIZON = 10080.0
TS = [1440.0, 4320.0, 8640.0]      # 1 d / 3 d / 6 d（t=7 d 时因随访被截断而无人在险，不可算 IPCW 口径）

# ---------------------------------------------------------------- 队列重建
def load_cohorts():
    """返回 (mimic_df, eicu_df, T_m, S_m, T_e, S_e)；行序与 _ml/model_*.csv 一致。"""
    M = pd.read_csv(r"D:/SEEDANCE/EICU/csv/mimic_s3b.csv", low_memory=False)
    E = pd.read_csv(r"D:/SEEDANCE/EICU/csv/eicu_s3.csv", low_memory=False)
    m = M[M["cr_base24"].notna()].reset_index(drop=True).copy()
    e = E[E["cr_base24"].notna()].reset_index(drop=True).copy()

    t0 = pd.to_datetime(m["t0"], errors="coerce")
    dis = pd.to_datetime(m["dischtime"], errors="coerce")
    dea = pd.to_datetime(m["deathtime"], errors="coerce")
    fu = ((dis - t0).dt.total_seconds() / 60.0).clip(lower=1, upper=HORIZON)
    fu = fu.where(fu.notna(), HORIZON).values
    aki = pd.to_numeric(m["aki_off"], errors="coerce").values
    aki_f = ((~np.isnan(aki)) & (aki <= fu))
    death_f = (pd.to_numeric(m["hospital_expire_flag"], errors="coerce") == 1).values
    dt = ((dea - t0).dt.total_seconds() / 60.0).values
    death_f = death_f & (~np.isnan(dt)) & (dt >= 0) & (dt <= fu)

    T_m, S_m = build_T(len(m), fu, aki, aki_f, dt, death_f)

    t0e = pd.to_numeric(e["t0"], errors="coerce").values
    fue = np.clip(pd.to_numeric(e["hospitaldischargeoffset"], errors="coerce").values - t0e, None, HORIZON)
    fue = np.where(np.isnan(fue) | (fue <= 0), HORIZON, fue)
    akie = pd.to_numeric(e["aki_off"], errors="coerce").values
    akie_f = ((~np.isnan(akie)) & (akie <= fue))
    deathe_f = (e["hospitaldischargestatus"].astype(str).str.strip().str.lower() == "expired").values
    dte = np.where(deathe_f, fue, np.nan)
    T_e, S_e = build_T(len(e), fue, akie, akie_f, dte, deathe_f)
    return m, e, T_m, S_m, T_e, S_e

def build_T(n, fu, aki_t, aki_flag, death_t, death_flag):
    T = np.minimum(np.asarray(fu, float), HORIZON)
    st = np.zeros(n, int)
    d = np.asarray(death_flag, bool) & ~np.isnan(death_t)
    T[d] = np.minimum(T[d], death_t[d]); st[d] = 2
    a = np.asarray(aki_flag, bool) & ~np.isnan(aki_t) & (aki_t <= HORIZON)
    sel = a & (aki_t < T)
    T[sel] = aki_t[sel]; st[sel] = 1
    return T, st

def split_indices(m, seed=123):
    from sklearn.model_selection import train_test_split
    mm = pd.read_csv(r"D:/SEEDANCE/CN 2/_ml/model_mimic.csv")
    assert len(mm) == len(m)
    tr, va = train_test_split(np.arange(len(mm)), test_size=0.3, random_state=seed,
                              stratify=mm["aki_7d"].values)
    return tr, va

def load_preds():
    pv = pd.read_csv(r"D:/SEEDANCE/CN 2/_ml/pred_val.csv")
    pe = pd.read_csv(r"D:/SEEDANCE/CN 2/_ml/pred_eicu.csv")
    return pv, pe

# ---------------------------------------------------------------- 基础估计量
def km_censoring(T, S):
    """删失分布 G(t)：仅 cause 0 为删失事件（向量化 KM）。"""
    order = np.argsort(T, kind="mergesort"); Ts, Ss = T[order], S[order]
    n = len(T)
    uniq, inv, _ = np.unique(Ts, return_inverse=True, return_counts=True)
    first = np.searchsorted(Ts, uniq, side="left")
    n_risk = n - first
    d_c = np.bincount(inv, weights=(Ss == 0).astype(float), minlength=len(uniq))
    f = np.where(d_c > 0, 1.0 - d_c / np.maximum(n_risk, 1), 1.0)
    times = np.concatenate([[0.0], uniq]); vals = np.concatenate([[1.0], np.cumprod(f)])
    def G(t):
        t = np.asarray(t, float)
        j = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(vals) - 1)
        return np.maximum(vals[j], 1e-6)
    return G

def aj_cif(T, S, t=HORIZON, cause=1):
    """Aalen–Johansen 累积发生函数（向量化）。"""
    order = np.argsort(T, kind="mergesort"); Ts, Ss = T[order], S[order]
    uniq, inv, _ = np.unique(Ts, return_inverse=True, return_counts=True)
    first = np.searchsorted(Ts, uniq, side="left")
    n_risk = np.maximum(len(T) - first, 1)
    d_c = np.bincount(inv, weights=(Ss == cause).astype(float), minlength=len(uniq))
    d_all = np.bincount(inv, weights=(Ss != 0).astype(float), minlength=len(uniq))
    s_all = np.concatenate([[1.0], np.cumprod(np.where(d_all > 0, 1.0 - d_all / n_risk, 1.0))[:-1]])
    cif = np.cumsum(s_all * d_c / n_risk)
    j = np.searchsorted(uniq, t, side="right") - 1
    return float(cif[j]) if j >= 0 else 0.0

def uno_c_slow(T, D, risk, G):
    """参照实现（O(n²)，仅用于交叉校验）。"""
    num = den = 0.0
    w = 1.0 / (G(T) ** 2)
    for i in np.where(D == 1)[0]:
        later = T > T[i]
        if not later.any():
            continue
        rj = risk[later]
        num += w[i] * ((risk[i] > rj).sum() + 0.5 * (risk[i] == rj).sum())
        den += w[i] * later.sum()
    return num / den if den > 0 else np.nan

def uno_c(T, D, risk, G):
    """Uno's C（IPCW，权重 1/G(T_i)²，事件为 cause 1）；BIT 加速，O(n log n)。

    一致性对 C = Σ_i w_i I(T_i<T_j) I(r_i>r_j) / Σ_i w_i I(T_i<T_j)（i 取事件者）。
    遍历按风险**升序**：已插入者风险严格更低，其中 T_j > T_i 者即一致对；
    同风险组内按 T 序给 0.5 权重；分母用全样本 T_j > T_i 的计数（与风险无关）。
    """
    n = len(T)
    w = 1.0 / (G(T) ** 2)
    oT = np.argsort(T, kind="mergesort")
    rank = np.empty(n, int); rank[oT] = np.arange(n)
    Ts = T[oT]
    uniq_T, first = np.unique(Ts, return_index=True)
    last = np.append(first[1:], n) - 1
    lut = dict(zip(uniq_T.tolist(), last.tolist()))
    rmax = np.array([lut[v] for v in T.tolist()], dtype=int)
    ev = np.where(D == 1)[0]
    if ev.size == 0:
        return np.nan
    den = float((w[ev] * (n - 1 - rmax[ev])).sum())
    order = np.argsort(risk, kind="mergesort")      # 升序
    bit = np.zeros(n + 2)
    def upd(i):
        i += 1
        while i <= n + 1:
            bit[i] += 1.0; i += i & (-i)
    def q(i):
        i += 1; s = 0.0
        while i > 0:
            s += bit[i]; i -= i & (-i)
        return s
    num = 0.0
    i = 0
    while i < n:
        r0 = risk[order[i]]
        j = i
        while j < n and risk[order[j]] == r0:
            j += 1
        grp = order[i: j]
        for g in grp:
            if D[g] == 1:
                c = i - q(rmax[g])
                if c > 0:
                    num += w[g] * c
        if len(grp) > 1:
            Tg = T[grp]
            for k in range(len(grp)):
                g = grp[k]
                if D[g] == 1:
                    c = int((Tg > Tg[k]).sum())
                    if c:
                        num += 0.5 * w[g] * c
        for g in grp:
            upd(rank[g])
        i = j
    return num / den if den > 0 else np.nan

def auc_t(T, S, risk, t, G, cause=1):
    """IPCW 时间依赖 AUC（t 前发生竞争事件者权重 0）。"""
    case = (T <= t) & (S == cause)
    ctrl = T > t
    if case.sum() == 0 or ctrl.sum() == 0:
        return np.nan
    rc = np.sort(risk[ctrl]); a = 1.0 / G(T[case])
    num = den = 0.0
    for pi, wi in zip(risk[case], a):
        lo = np.searchsorted(rc, pi, side="left"); hi = np.searchsorted(rc, pi, side="right")
        num += wi * (lo + 0.5 * (hi - lo)); den += wi * rc.size
    return num / den

def brier_t(T, S, risk, t, G, cause=1):
    """IPCW Brier（竞争风险口径）。"""
    w = np.zeros(len(T)); y = np.zeros(len(T))
    case = (T <= t) & (S == cause); free = T > t
    w[case] = 1.0 / G(T[case]); y[case] = 1.0
    w[free] = 1.0 / float(G(t)); y[free] = 0.0
    m = w > 0
    return float((w[m] * (risk[m] - y[m]) ** 2).sum() / w[m].sum())

def nri_idi(y, p_new, p_old):
    """连续 NRI 与 IDI（二分类事件口径）。"""
    ev = y == 1; ne = y == 0
    d = p_new - p_old
    up, dn = d > 0, d < 0
    nri = ((up[ev].mean() if ev.sum() else 0) - (dn[ev].mean() if ev.sum() else 0)
           + (dn[ne].mean() if ne.sum() else 0) - (up[ne].mean() if ne.sum() else 0))
    idi = ((p_new[ev].mean() - p_old[ev].mean()) if ev.sum() else 0) - \
          ((p_new[ne].mean() - p_old[ne].mean()) if ne.sum() else 0)
    return float(nri), float(idi)

def dca(T, S, risk, thresholds, cause=1):
    """净获益（病例=7 d 内 AKI；其余为非病例）。"""
    y = ((T <= HORIZON) & (S == cause)).astype(int)
    n = len(y)
    out = []
    for pt in thresholds:
        pos = risk >= pt
        out.append(float((pos & (y == 1)).sum() / n - (pos & (y == 0)).sum() / n * (pt / (1 - pt))))
    return np.array(out)

def auc_binary(y, risk):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, risk))

def brier_binary(y, risk):
    y = np.asarray(y, float); risk = np.asarray(risk, float)
    return float(np.mean((risk - y) ** 2))

def auc_excl_death(T, S, risk, t=HORIZON):
    """竞争风险口径 AUC：病例=7 d 内 AKI；对照=未见 AKI 者（含出院截断）；7 d 前死亡者排除。"""
    keep = S != 2
    y = ((T <= t) & (S == 1)).astype(int)[keep]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    return auc_binary(y, np.asarray(risk)[keep])

def brier_excl_death(T, S, risk, t=HORIZON):
    keep = S != 2
    y = ((T <= t) & (S == 1)).astype(int)[keep]
    return brier_binary(y, np.asarray(risk)[keep])

def logrank_like_oe(T, S, risk, t=HORIZON, q=4):
    """预测风险四分位分层的观测 AKI 发生率（全样本口径）与层间梯度。"""
    risk = np.asarray(risk, float)
    y = ((T <= t) & (S == 1)).astype(int)
    rank = pd.qcut(pd.Series(risk), q, labels=False, duplicates="drop").values
    out = []
    for g in np.unique(rank[~pd.isna(rank)]):
        m = (rank == g)
        out.append(dict(quartile=int(g) + 1, n=int(m.sum()),
                        pred_mean=float(risk[m].mean()), obs_rate=float(y[m].mean())))
    return pd.DataFrame(out)

def metrics(T, S, risk, with_binary=True):
    """一次性返回全部判别力指标（三套口径并列）。"""
    G = km_censoring(T, S)
    D = (S == 1).astype(int)
    y = ((T <= HORIZON) & (S == 1)).astype(int)
    r = {"C_uno": uno_c(T, D, risk, G),
         "AUC_bin": auc_binary(y, risk),
         "Brier_bin": brier_binary(y, risk),
         "AUC_exd": auc_excl_death(T, S, risk),
         "Brier_exd": brier_excl_death(T, S, risk),
         "CIF_obs_7d": aj_cif(T, S, HORIZON, 1),
         "O_E_7d": aj_cif(T, S, HORIZON, 1) / max(float(np.mean(risk)), 1e-9)}
    if with_binary:
        r["C_binary"] = r["AUC_bin"]
    for t in TS:
        r["AUC_t%d" % int(t / 1440)] = auc_t(T, S, risk, t, G)
        r["Brier_t%d" % int(t / 1440)] = brier_t(T, S, risk, t, G)
    return r
