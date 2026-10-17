# -*- coding: utf-8 -*-
"""test_app.py —— 用 Streamlit 官方 AppTest 无头运行 app.py，验证无异常且能出结果"""
import os, sys
from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(os.path.abspath(__file__))
at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=90).run()

print("异常数：%d" % len(at.exception))
for e in at.exception:
    print("  ❗", e.value if hasattr(e, "value") else e)
assert not at.exception, "app.py 运行出现异常"

print("标题：", at.title[0].value if at.title else "（无）")
print("输入控件数：number_input=%d, checkbox=%d" % (len(at.number_input), len(at.checkbox)))
assert len(at.number_input) + len(at.checkbox) >= 29, "输入控件少于 29 个"

# 点击"计算 7 d AKI 风险"按钮，验证输出
btns = [b for b in at.button]
print("按钮数：%d" % len(btns))
btns[0].click().run()
assert not at.exception, "点击计算后出现异常"
metrics = [m.value for m in at.metric]
print("输出指标：", metrics)
assert any("%" in str(v) for v in metrics), "未输出风险百分比"
print("\nAPP TEST：PASS")
