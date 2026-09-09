# AlphaRisk-Engine 📈

*[English](README)*

一个端到端的多因子组合风险分析平台：一条抓取真实市场数据并拟合六因子风险模型的数据管道，加上一个把模型结果变成完整风险分解工作台的 Streamlit 交互终端，支持任意自定义组合。

## 🚀 这个项目做什么

**数据管道**（[main.py](main.py)）—— 拉取实时的标普 500 成分股列表和每日行情（`yfinance`），清洗归一化后，再拉取 Fama-French/Carhart 多因子收益率序列，对每只股票用手写的线性代数引擎批量求解六因子 OLS 回归（市场、规模、价值、盈利能力、投资风格、动量），矩阵病态时自动回退到 Ridge（Tikhonov）正则化。所有结果都存进 SQLite，用的是纵表（narrow-table）schema，以后加新因子不需要改表结构。

**风险终端**（[app.py](app.py)）—— 一个 Streamlit 应用，你可以搜索/添加股票、按板块批量加仓，或者直接用默认持仓组合，实时得到：

- **持仓与暴露**：每只股票的因子载荷、板块构成、权重编辑器。
- **主动风险**（相对 SPY 基准）：跟踪误差、主动因子偏离（Δβ）、"风格押注 vs 选股集中"的归因、因子/特异风险占比。
- **绝对风险**：年化波动率、系统性/特异性风险占比、组合整体因子暴露（`B_p = Bᵀw`）、总方差分解，分成三个 tab：
  - *因子暴露*：组合整体六因子画像 + 逐个持仓的因子载荷矩阵。
  - *风险分解*：方差分解饼图、**边际风险贡献**（用完整协方差矩阵 `Σ = B Σ_F Bᵀ + D` 做 Euler 分解，而不只是看孤立的 β——能揪出那些因为跟组合里其他持仓高度相关、实际风险贡献远超其仓位占比的股票）、**VaR / 条件 VaR**（参数法高斯 VaR 和预期损失，置信度和持有期可调）。
  - *相关性*：基于同一个完整协方差矩阵算出的持仓两两相关性热力图，附带自动生成的"相关性最高/最低那一对"文字解读。
- **规则引擎生成的解读文字**（`src/risk_narrator.py`）：每张图都配一段大白话诊断，不接 LLM，纯确定性、可审计的文本层，解释每个数字是什么意思、为什么重要，颜色跟图表一一对应。

## 🏗️ 架构

```text
yfinance / Fama-French  ──▶  MarketDataFetcher  ──▶  FinancialDataCleaner
                                                            │
                                                            ▼
                                                  FactorExposureEngine (OLS 回归)
                                                            │
                                                            ▼
                                                     RiskDatabase (SQLite)
                                                            │
                                                            ▼
                              app.py (Streamlit)  ◀──  RiskDecompositionEngine
                                     │                  （方差分解 / MCTR / VaR / 相关性）
                                     ▼
                              risk_narrator.py（大白话诊断文字）
```

## 🚀 核心特性与工程实践

- **一切风险指标都源自同一套线性代数**：因子方差、边际风险贡献、参数法 VaR/CVaR、持仓相关性——全部从同一个完整协方差矩阵 `Σ = B Σ_F Bᵀ + D` 推导，而不是用孤立的单股 β 走捷径。
- **纵表 schema**：因子收益率和因子暴露都按行存储（`factor_name` 是一行而不是一列），以后加第七个因子不需要动表结构。
- **防御性数值计算**：手写的 OLS 求解器会检查 `XᵀX` 的条件数，矩阵病态时自动切到 Ridge 正则化，而不是悄悄返回一堆算错的 β。
- **规则引擎解读，不是 LLM**：`risk_narrator.py` 是纯离线、确定性、可审计的文本层——每个数字都配一个参照系（"这算高还是低"）和一句大白话解释，没有推理延迟，也没有幻觉风险。
- **模块化管道**：`MarketDataFetcher → FinancialDataCleaner → FactorExposureEngine → RiskDatabase → RiskDecompositionEngine`，每一环都能独立测试。

## 🛠️ 技术栈

- **语言**：Python 3.10+
- **数据处理**：Pandas、NumPy、SciPy
- **界面**：Streamlit、Plotly
- **数据库**：SQLite（`factor_risk.db`）—— 纵表 schema，可平滑迁移到 PostgreSQL
- **测试**：Pytest
- **数据源**：`yfinance`（雅虎财经 API）、Fama-French 因子数据库

## 📂 项目结构

```text
AlphaRisk-Engine/
├── main.py                        # ETL + 因子回归管道入口
├── app.py                         # Streamlit 风险终端（持仓 / 主动风险 / 绝对风险）
├── requirements.txt
├── src/
│   ├── data_fetcher.py            # yfinance + Fama-French API 对接
│   ├── cleaner.py                 # 原始行情清洗与归一化
│   ├── factor_model.py            # FactorExposureEngine：正规方程 OLS + Ridge 回退
│   ├── db_client.py               # RiskDatabase：SQLite 纵表 schema 与持久化
│   ├── risk_decomposition.py      # RiskDecompositionEngine：方差分解 / MCTR / VaR-CVaR / 相关性
│   └── risk_narrator.py           # 逐图配套的规则引擎大白话诊断
├── tests/
│   └── test_risk_math.py          # 风险引擎的数学恒等式测试
└── factor_risk.db                 # SQLite 数据库（由 main.py 生成）
```

## ▶️ 快速开始

```bash
pip install -r requirements.txt

# 1. 跑一遍 ETL + 因子回归管道（生成 factor_risk.db）
python main.py

# 2. 启动风险终端
streamlit run app.py
```

## 🧪 测试

```bash
pytest tests/ -v
```

覆盖了 OLS 求解器能否精确复原已知的合成参数，以及风险引擎依赖的几条数学恒等式：`factor_variance + specific_variance == total_variance`、`Σ MCTR_i == total_annualized_vol`、VaR/CVaR 的大小关系与 √time 缩放，以及相关性确实来自完整协方差结构而不只是 β 相同就判定强相关。

## ⚠️ 已知局限

- **参数法 VaR/CVaR** 假设每日收益率服从零均值正态分布——这是短持有期下的简化假设，不能替代历史模拟法或蒙特卡洛模拟在肥尾组合上的表现。
- `api_server.py`（为未来外部前端准备的 FastAPI 接口）和 `src/visualizer.py`（静态 HTML 报告导出）还处于早期/实验阶段，尚未接入主 Streamlit 流程。
