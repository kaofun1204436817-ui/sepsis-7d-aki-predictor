# 脓毒症患者 7 d 急性肾损伤风险预测模型（双库开发与外部验证）

基于 **MIMIC-IV 3.1**（开发与内部验证）与 **eICU-CRD 2.0**（外部验证）构建的 **landmark 24 h** 设计
风险预测模型，用于预测脓毒症患者 **landmark（t₀＋24 h）后 7 d 内**发生急性肾损伤（AKI）的风险。

> ⚠️ **研究用途（research use only）**：本项目及配套工具仅用于学术研究与方法学演示，不构成临床建议，
> 不得用于临床诊疗决策。

---

## 一、研究概要

| 项 | 内容 |
|---|---|
| 研究类型 | 回顾性队列研究（公开重症数据库二次分析） |
| 人群 | Sepsis-3 成人 ICU 患者，CRRT 与 AKI 事件按 KDIGO 肌酐标准判定 |
| **预测时点** | **t₀ ＋ 24 h（landmark）**：变量取自 [t₀, t₀＋24 h] 窗口，排除 landmark 前已发生 AKI 或死亡者 |
| 结局 | landmark 后、7 d 内（自 t₀ 起算）发生 AKI；死亡为竞争事件 |
| 队列规模 | MIMIC-IV 28,090 例 → 可入组 28,090（训练 19,663／内部验证 8,427）；eICU-CRD 可入组 26,333 |
| 事件数 | 训练 3,402／内部验证 1,458／外部 5,315 |
| 变量筛选 | 单因素（P<0.05）∩ LASSO（10 折 CV, λ.1se）∩ Boruta → **29 变量** |
| 模型矩阵 | LR、RF、XGBoost、LightGBM、SVM、MLP、GaussianNB、Bagging（GridSearchCV, cv=5, roc_auc） |
| 主模型 | **XGBoost**（内部验证 AUC 最高） |

**主要性能（29 变量集）**

| 指标 | 内部验证集（MIMIC-IV, n=8,427） | 外部验证集（eICU-CRD, n=26,333） |
|---|---|---|
| AUC（95%CI） | **0.768（0.754–0.782）** | **0.705（0.697–0.712）** |
| PR-AUC | 0.427 | 0.384 |
| Brier 评分 | 0.122 | 0.149 |
| Uno's C（竞争风险） | 0.738 | 0.676 |
| 竞争风险口径 AUC | 0.774 | 0.725 |
| 校准斜率／O/E | 1.03／1.11 | 0.76／1.51（**Platt 再校准后 1.01／1.15**） |

**时间验证**（MIMIC-IV 官方 `anchor_year_group`：2008–2013 训练 → 2014–2022 验证）
AUC 0.740–0.747，校准斜率（RF）1.099。

**风险分层（内部验证集，四分位观测 AKI 率）**：Q1 4.3% → Q2 10.0% → Q3 17.2% → **Q4 37.8%**

**决策阈值（内部验证集）**

| 规则 | 阈值 | 敏感度 | 特异度 | PPV | NPV |
|---|---:|---:|---:|---:|---:|
| 规则排除（敏感度≈90%） | 0.089 | 0.900 | 0.404 | 0.240 | **0.951** |
| 固定阈值 ≥20% | 0.200 | 0.629 | 0.761 | 0.355 | 0.907 |
| 规则纳入（特异度≈90%） | 0.308 | 0.418 | 0.900 | **0.467** | 0.881 |

---

## 二、目录结构

```
code_release/
├── README.md                  # 本文件
├── requirements.txt           # 依赖
├── LICENSE                    # MIT（代码）；数据遵循 PhysioNet DUA
├── app.py                     # Streamlit 在线预测工具
├── predictor.py               # 推理模块（模型加载、风险计算、分层、局部 SHAP）
├── feature_defaults.json      # 29 变量元数据（名称/单位/中位数/P5/P95/取值范围）
├── model_artifact/
│   └── best_model_xgb.pkl     # 主模型（XGBoost）＋变量集
├── test_predictor.py          # 推理链路自检
├── test_app.py                # Streamlit 应用无头自检（官方 AppTest）
├── 特征字典.md                # 变量定义、窗口、缺失率与中文名
└── analysis/                  # 全流程可复现脚本
    ├── gen_n1_sql.py / n1_extract_mimic.sql     # MIMIC 变量提取（SQL 生成 + 执行）
    ├── gen_n1_eicu_sql.py / n1_extract_eicu.sql # eICU 变量提取
    ├── n1_build_dataset.py / n1_build_eicu.py   # landmark 队列构建
    ├── n2_prepare.py / n2_screen.R / n2_intersect.py   # 第一轮变量筛选（41 变量池）
    ├── n3_models.py                             # 第一轮 8 模型矩阵
    ├── n4_build_features.py                     # 时间趋势/派生特征（26 个）
    ├── n4_screen.py / n4_screen.R / n4_intersect.py    # 第二轮筛选（67 候选 → 29）
    ├── n4_models.py                             # 最终 8 模型矩阵 + 外部再校准 + 阈值分析
    ├── n5_analysis.py / n5_cc_fix.py            # 时间验证/稳健性/亚组/校准曲线
    └── n6_shap.py                               # SHAP 可解释性
```

## 三、快速开始

### 3.0 输出目录配置

分析脚本默认把结果写入 `analysis/outputs/`；可用环境变量 CN2_OUT 覆盖：

```bash
# Windows (cmd)
set CN2_OUT=D://your//output//dir/n/n# Linux / macOS
export CN2_OUT=/path/to/outputs
```

### 3.1 在线工具

在线演示（无需安装）：https://sepsis-7d-aki-predictor.streamlit.app

> 仅供研究演示，不用于临床决策。若页面提示应用休眠，首次打开可能需要数十秒唤醒。

本地运行：

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开后：左侧填写 29 个变量（默认值为训练集中位数，可直接体验），点击"计算 7 d AKI 风险"，
得到预测风险、风险分层、规则判定与单例 SHAP 贡献。

**自检**：

```bash
python test_predictor.py     # 推理链路（默认/高危/低危/缺失补齐/局部解释）
python test_app.py           # Streamlit 应用无头运行（AppTest）
```

### 3.2 复现分析流程

本项目分析依赖两个公开数据库，**须自行申请并本地建库**（本项目使用 PostgreSQL 17）：

1. 申请数据：MIMIC-IV 3.1 与 eICU-CRD 2.0（https://physionet.org，需通过 CITI 培训并签署 DUA）；
2. 建库后按 `analysis/gen_n1_sql.py`、`gen_n1_eicu_sql.py` 生成 SQL 并执行，导出窗口变量；
3. 依次运行 `n1_*` → `n2_*` → `n3_*` → `n4_*` → `n5_*` → `n6_*`（Python 与 R 混合，随机种子固定为 123）。

> 脚本内的绝对路径请按本地环境修改；`analysis/` 中的 SQL 与 Python 脚本不含任何患者数据。

## 四、数据可得性

本研究使用 MIMIC-IV 3.1 与 eICU-CRD 2.0（PhysioNet 公开数据库）。**原始数据不可再分发**，
研究者需自行申请（https://physionet.org/content/mimiciv/ 与 https://physionet.org/content/eicu-crd/）。
本仓库**不含任何患者级数据**，仅包含代码、模型参数与变量元数据（中位数/P5/P95 等聚合统计）。

## 五、报告规范

分析流程遵循 **TRIPOD**（预测模型报告规范）与 **STROBE**（观察性研究）声明；
模型构建全程遵守"训练集内完成插补/筛选/调参、验证集与外部集仅用于评价"的防泄漏原则。

## 六、已知局限

1. 变量集受两库可比性约束——为保持外部验证可比，剔除了 eICU 缺失 >50% 的 6 个变量（乳酸、尿量、体温、P/F 等）；
   在 MIMIC 单库中纳入这些变量可再提升约 **0.02 AUC**（0.757→0.781），该差距源于**数据可得性**而非建模方法；
2. 外部验证存在系统性低估（O/E 1.51），Platt 再校准后降至 1.15（校准斜率 0.76→1.01）；
3. SOFA≥8 的重症亚组判别力偏低（AUC 0.688）；
4. 回顾性二次分析，需多中心前瞻队列进一步验证。

## 七、引用

如使用本代码或模型，请引用对应论文（待发表）与数据库原始文献：

- Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Sci Data.* 2023;10(1):1.
- Pollard TJ, et al. The eICU Collaborative Research Database. *Sci Data.* 2018;5:180178.
- Collins GS, et al. TRIPOD statement. *BMJ.* 2015;350:g7594.

## 八、许可

代码以 **MIT 许可证**发布（见 `LICENSE`）。数据库使用受 PhysioNet 数据使用协议约束。
