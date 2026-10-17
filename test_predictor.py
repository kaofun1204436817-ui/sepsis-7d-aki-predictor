# -*- coding: utf-8 -*-
"""test_predictor.py —— 推理链路自检（不依赖 Streamlit 服务）"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import predict_risk, stratify, local_contributions, META_MAP, FEATURES

ok = True
print("变量数：%d" % len(FEATURES))

# 1) 默认值（中位数）预测
defaults = {f: (META_MAP[f].get("median", 0.0) if META_MAP[f]["type"] == "continuous" else 0)
            for f in FEATURES}
r0 = predict_risk(defaults)
print("默认（中位数）预测风险：%.4f" % r0)
ok &= 0.0 <= r0 <= 1.0

# 2) 高危构造（SOFA 高、基线肌酐高、肌酐峰值/基线比高、机械通气、心衰、CKD）
hi = dict(defaults)
hi.update({"sofa": 12, "cr_base": 3.0, "cr_ratio": 1.4, "cr_delta": 1.0, "vent": 1, "chf": 1, "ckd": 1,
           "bun_cr_ratio": 30, "rr_post_max": 32, "hr_post_max": 130, "sbp_delta": 60})
r_hi = predict_risk(hi)
# 3) 低危构造
lo = dict(defaults)
lo.update({"sofa": 3, "cr_base": 0.7, "cr_ratio": 1.0, "cr_delta": 0.05, "vent": 0, "chf": 0, "ckd": 0,
           "bun_cr_ratio": 10, "rr_post_max": 16, "hr_post_max": 85, "sbp_delta": 15})
r_lo = predict_risk(lo)
print("高危构造：%.4f（分层 %s）｜低危构造：%.4f（分层 %s）"
      % (r_hi, stratify(r_hi)["band"], r_lo, stratify(r_lo)["band"]))
ok &= r_hi > r_lo and r_hi <= 1.0 and r_lo >= 0.0

# 4) 缺失项自动以中位数补齐
partial = {"sofa": 10, "cr_base": 2.0}
print("仅给 2 个变量：%.4f" % predict_risk(partial))

# 5) 局部解释
c = local_contributions(hi, top=5)
print("高危构造的 top-5 贡献：", c)
ok &= len(c) == 5

# 6) 与训练集整体风险量级对照（内部验证集平均预测约 0.17–0.20）
print("\n自检结果：", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
