# -*- coding: utf-8 -*-
"""
predictor.py —— CN2 7 d AKI 风险预测（推理模块，供 Streamlit 与脚本共用）
⚠️ 研究用途（research use only）：不得用于临床决策。
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))

# 阈值来自 N4 阈值分析（内部验证集）：
#   规则排除（敏感度≈90%）：0.089；规则纳入（特异度≈90%）：0.308；中高危分层：0.20
THR_RULE_OUT = 0.089
THR_MID = 0.200
THR_RULE_IN = 0.308

def load_artifacts(model_path=None, meta_path=None):
    model_path = model_path or os.path.join(HERE, "model_artifact", "best_model_xgb.pkl")
    meta_path = meta_path or os.path.join(HERE, "feature_defaults.json")
    bundle = joblib.load(model_path)
    meta = json.load(open(meta_path, encoding="utf-8"))
    return bundle, meta

BUNDLE, META = load_artifacts()
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["features"]
META_MAP = META["meta"]

def predict_risk(values: dict) -> float:
    """values：{变量名: 数值}；缺失项以训练集中位数（或 0）补齐；返回 7 d AKI 预测风险。"""
    row = {}
    for f in FEATURES:
        v = values.get(f, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            m = META_MAP.get(f, {})
            v = m.get("median", 0.0) if m.get("type") == "continuous" else 0
        row[f] = float(v)
    X = pd.DataFrame([row])[FEATURES]
    return float(MODEL.predict_proba(X)[:, 1][0])

def stratify(risk: float) -> dict:
    """按预设阈值给出分层与规则判定。"""
    if risk >= THR_RULE_IN:
        band, action = "高危", "建议纳入加强监测/预防策略（规则纳入阈值 0.308）"
    elif risk >= THR_MID:
        band, action = "中危", "建议标准监测，按临床判断决定是否加强"
    elif risk >= THR_RULE_OUT:
        band, action = "低-中危", "常规监测"
    else:
        band, action = "低危", "可考虑规则排除（敏感度≈90% 阈值 0.089，NPV≈0.95）"
    return {"risk": risk, "band": band, "action": action,
            "thr_rule_out": THR_RULE_OUT, "thr_mid": THR_MID, "thr_rule_in": THR_RULE_IN}

def local_contributions(values: dict, top=8):
    """返回单例 TreeSHAP 贡献（用于解释）。依赖 xgboost 原生 pred_contribs。"""
    try:
        from xgboost import DMatrix
        row = {}
        for f in FEATURES:
            v = values.get(f, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                m = META_MAP.get(f, {})
                v = m.get("median", 0.0) if m.get("type") == "continuous" else 0
            row[f] = float(v)
        X = pd.DataFrame([row])[FEATURES]
        sv = MODEL.get_booster().predict(DMatrix(X, feature_names=FEATURES), pred_contribs=True)[0][:-1]
        items = sorted(zip(FEATURES, sv), key=lambda t: -abs(t[1]))[:top]
        return [(META_MAP.get(f, {}).get("label", f), float(s)) for f, s in items]
    except Exception:
        return []
