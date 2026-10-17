# -*- coding: utf-8 -*-
"""n4_intersect.py —— N2 第三步：三方法交集（单因素 ∩ LASSO λ.1se ∩ Boruta）"""
import os
import pandas as pd

OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
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

uni = pd.read_csv(os.path.join(OUT, "n4_univariate.csv"))
las = pd.read_csv(os.path.join(OUT, "n4_lasso.csv"))
bor = pd.read_csv(os.path.join(OUT, "n4_boruta.csv"))

U = set(uni.loc[uni["uni_sig"] == True, "var"])            # noqa: E712
L = set(las.loc[las["lasso_1se"] == True, "var"])          # noqa: E712
Lmin = set(las.loc[las["lasso_min"] == True, "var"])       # noqa: E712
BC = set(bor.loc[bor["decision"] == "Confirmed", "var"])
BT = set(bor.loc[bor["decision"] == "Tentative", "var"])

allv = sorted(set(uni["var"]) | L | BC)
rows = []
for v in allv:
    pin = (v in U) and (v in L) and (v in BC)
    rows.append(dict(var=v, label=LABELS.get(v, v), in_uni=v in U, in_lasso_1se=v in L,
                     in_lasso_min=v in Lmin, in_boruta_conf=v in BC, in_boruta_tent=v in BT,
                     primary_intersection=pin,
                     decision=("三方法交集" if pin else
                               ("LASSO+单因素" if (v in U and v in L) else
                                ("Boruta+单因素" if (v in U and v in BC) else
                                 ("单因素+LASSO(min)" if (v in U and v in Lmin) else "未入选"))))))
sel = pd.DataFrame(rows).sort_values(["primary_intersection", "var"], ascending=[False, True])
sel.to_csv(os.path.join(OUT, "n4_selected_variables.csv"), index=False, encoding="utf-8-sig")

PRIM = sorted(sel.loc[sel["primary_intersection"], "var"])
print("候选变量池: %d" % len(allv))
print("单因素显著(P<0.05): %d ｜ LASSO λ.1se: %d ｜ LASSO λ.min: %d ｜ Boruta Confirmed: %d ｜ Tentative: %d"
      % (len(U), len(L), len(Lmin), len(BC), len(BT)))
print("\n★ 三方法交集（%d 个）:" % len(PRIM))
for v in PRIM:
    print("   %-16s %s" % (v, LABELS.get(v, v)))
print("\n两两重合度：单因素∩LASSO=%d ｜ 单因素∩Boruta=%d ｜ LASSO∩Boruta=%d"
      % (len(U & L), len(U & BC), len(L & BC)))
print("\n边界情况（未进交集但值得记录）：")
for v in sorted(set(U | L | BC) - set(PRIM)):
    tags = []
    if v in U: tags.append("单因素")
    if v in L: tags.append("LASSO1se")
    if v in Lmin: tags.append("LASSOmin")
    if v in BC: tags.append("Boruta")
    if v in BT: tags.append("Tentative")
    print("   %-16s %s" % (LABELS.get(v, v), "+".join(tags)))
print("\nSAVED: n4_selected_variables.csv")
