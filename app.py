# -*- coding: utf-8 -*-
"""
app.py —— 脓毒症患者 7 d 急性肾损伤风险预测工具（Streamlit）
⚠️ 研究用途，不得用于临床决策。
运行：streamlit run app.py
"""
import streamlit as st
import pandas as pd
from predictor import predict_risk, stratify, local_contributions, META_MAP, FEATURES

st.set_page_config(page_title="7 d AKI 风险预测（研究工具）", page_icon="🩺", layout="wide")

st.title("脓毒症患者 7 d 急性肾损伤风险预测（研究工具）")
st.caption("基于 MIMIC-IV 开发、eICU-CRD 外部验证的双库模型（landmark 24 h 设计，29 变量，XGBoost）"
           "　｜　外部验证 AUC 0.705（0.697–0.712），内部 0.768（0.754–0.782）")
st.warning("⚠️ 本工具仅用于研究与方法学演示，**不得用于临床诊疗决策**。预测值不构成任何医疗建议。")

GROUPS = {
    "人口学与疾病严重度": ["age", "sofa", "cr_base"],
    "基础疾病与治疗": ["chf", "ckd", "cva", "dm", "liver", "vent"],
    "24 h 窗口实验室（最差值）": ["agap_post_max", "cl_post_min", "hb_post_min", "k_post_max",
                            "mchc_post_max", "mcv_post_max", "rdw_post_max", "wbc_post_max", "bun_cr_ratio"],
    "24 h 窗口实验室变化幅度（峰值−谷值）": ["agap_delta", "hb_delta", "hct_delta", "plt_delta", "cr_delta"],
    "肾功能轨迹": ["cr_ratio"],
    "24 h 窗口生命体征与波动": ["hr_post_max", "hr_delta", "rr_post_max", "sbp_delta", "sbp_rel_delta"],
}

values = {}
left, right = st.columns([3, 2])
with left:
    st.subheader("① 输入变量")
    for gname, keys in GROUPS.items():
        with st.expander(gname, expanded=(gname == "人口学与疾病严重度")):
            cols = st.columns(2)
            for i, k in enumerate(keys):
                m = META_MAP.get(k, {})
                label = m.get("label", k)
                unit = ("（%s）" % m["unit"]) if m.get("unit") else ""
                with cols[i % 2]:
                    if m.get("type") == "binary":
                        values[k] = 1 if st.checkbox("%s %s" % (label, unit), key=k) else 0
                    else:
                        values[k] = st.number_input(
                            "%s%s" % (label, unit), value=float(m.get("median", 0.0)),
                            min_value=float(m.get("lo", 0.0)), max_value=float(m.get("hi", 1e6)),
                            step=float(max((m.get("p95", 1) - m.get("p5", 0)) / 50.0, 0.01)), key=k)

with right:
    st.subheader("② 预测结果")
    if st.button("计算 7 d AKI 风险", type="primary"):
        r = predict_risk(values)
        s = stratify(r)
        st.metric("7 d AKI 预测风险", "%.1f%%" % (100 * r))
        color = {"高危": "🔴", "中危": "🟠", "低-中危": "🟡", "低危": "🟢"}[s["band"]]
        st.markdown("**风险分层：** %s %s" % (color, s["band"]))
        st.info(s["action"])
        st.markdown("| 阈值 | 取值 | 用途 |\n|---|---|---|\n"
                    "| 规则排除 | %.3f | 高敏感度（≈90%%），NPV≈0.95 |\n"
                    "| 中危分层 | %.3f | 固定阈值 ≥20%%，敏感度 0.63／特异度 0.76 |\n"
                    "| 规则纳入 | %.3f | 高特异度（≈90%%），PPV≈0.47 |"
                    % (s["thr_rule_out"], s["thr_mid"], s["thr_rule_in"]))
        contrib = local_contributions(values)
        if contrib:
            st.markdown("**本例主要贡献变量（TreeSHAP，正=推高风险）**")
            st.dataframe(pd.DataFrame(contrib, columns=["变量", "SHAP 值"]).round(4),
                         hide_index=True, width="stretch")
    else:
        st.write("填写左侧变量后点击计算。默认值为训练集**中位数**，可直接体验。")
        st.markdown("**变量清单（29 个）**：" + "、".join(META_MAP.get(f, {}).get("label", f) for f in FEATURES))

with st.expander("③ 模型性能摘要（详见论文）"):
    st.markdown("""
| 指标 | 内部验证集（MIMIC-IV, n=8,427） | 外部验证集（eICU-CRD, n=26,333） |
|---|---|---|
| AUC（95%CI） | 0.768（0.754–0.782） | 0.705（0.697–0.712） |
| Brier 评分 | 0.122 | 0.149 |
| 竞争风险口径 AUC | 0.774 | 0.725 |
| 校准斜率／O/E | 1.03／1.11 | 0.76／1.51（Platt 再校准后 1.01／1.15） |
| 时间验证（2008–2013 → 2014–2022） | AUC 0.740–0.747 | — |

**风险分层（内部验证集，四分位观测 AKI 率）**：Q1 4.3% → Q2 10.0% → Q3 17.2% → Q4 37.8%
""")
st.caption("模型与阈值文件：model_artifact/best_model_xgb.pkl、feature_defaults.json　｜　"
           "数据来自 MIMIC-IV 与 eICU-CRD（PhysioNet，需自行申请），本仓库不含任何患者数据。")
