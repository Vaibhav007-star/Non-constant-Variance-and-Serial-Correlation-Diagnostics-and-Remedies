# 📊 Econometric Diagnostics & Remedies: Non-constant Variance & Serial Correlation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.15%2B-blue.svg)](https://www.statsmodels.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **An enterprise-grade econometric research toolkit and interactive cockpit for detecting, diagnosing, and remedying Heteroscedasticity (non-constant variance) and Serial Correlation (autocorrelation) in regression modeling.**

---

## 📌 Executive Summary & Motivation

Under the classical **Gauss-Markov Assumptions**, Ordinary Least Squares (OLS) produces **Best Linear Unbiased Estimators (BLUE)**. However, real-world economic, financial, and cross-sectional data frequently violate two core tenets:

1. **Homoscedasticity:** $\text{Var}(\epsilon_i \mid X_i) = \sigma^2$ (Constant error variance).
2. **Error Independence:** $\text{Cov}(\epsilon_t, \epsilon_s \mid X) = 0$ for $t \neq s$ (No serial correlation).

### Consequences of Violations
| Property | Homoscedasticity Violated (Heteroscedasticity) | Error Independence Violated (Serial Correlation) |
| :--- | :--- | :--- |
| **OLS Coefficients ($\hat{\beta}$)** | **Unbiased & Consistent** | **Unbiased & Consistent** (with strictly exogenous regressors) |
| **OLS Efficiency** | **Inefficient** (No longer BLUE) | **Inefficient** (Higher variance than GLS) |
| **OLS Standard Errors** | **Biased** (Usually underestimated) | **Severely Biased Downward** |
| **Statistical Inference** | False positives ($t$-tests & $F$-tests invalid) | Spurious significance & misleading $R^2$ |

This repository delivers an end-to-end framework featuring **statistical hypothesis tests**, **publication-ready diagnostic visual galleries**, **robust covariance estimators (HC0–HC3, Newey-West HAC)**, **GLS/WLS**, **Prais-Winsten / Cochrane-Orcutt procedures**, and an **interactive Streamlit cockpit**.

---

## 🛠️ Complete Diagnostic & Remedy Architecture

```text
Non-constant-Variance-and-Serial-Correlation-Diagnostics-and-Remedies/
├── app/
│   ├── app.py                         # Interactive Streamlit Web Cockpit
│   └── styles.css                     # Dark glassmorphic design system
├── data/
│   ├── generate_datasets.py           # Canonical dataset generator
│   ├── macroeconomic_ts.csv          # Time-series (AR(1) & ARCH volatility)
│   ├── housing_prices.csv            # Cross-sectional (Fan heteroscedasticity)
│   └── corporate_wages.csv           # Human capital wage divergence
├── src/
│   ├── diagnostics/
│   │   ├── heteroscedasticity.py      # Breusch-Pagan, White, Goldfeld-Quandt
│   │   └── serial_correlation.py      # Durbin-Watson, Breusch-Godfrey LM, Ljung-Box
│   ├── remedies/
│   │   ├── robust_standard_errors.py  # White HC0-HC3, Newey-West HAC
│   │   ├── gls_wls.py                 # WLS & Feasible Generalized Least Squares
│   │   ├── time_series_remedies.py   # Cochrane-Orcutt, Prais-Winsten AR(1)
│   │   └── transformations.py        # Box-Cox & Natural Log Transforms
│   ├── visualizations/
│   │   └── diagnostic_plots.py       # Residuals vs Fitted, Scale-Location, ACF, Lag Scatter
│   └── pipeline/
│       └── econometric_engine.py      # Unified audit pipeline & scorecard generator
├── notebooks/
│   └── comprehensive_econometric_walkthrough.ipynb  # Interactive tutorial
├── tests/
│   ├── test_diagnostics.py            # Unit tests for statistical tests
│   └── test_remedies.py               # Unit tests for robust estimators
├── run_app.bat                        # Windows 1-click launcher
├── requirements.txt
└── README.md
```

---

## 🧪 Statistical Diagnostics Suite

### 1. Heteroscedasticity Tests
- **Breusch-Pagan Test ($LM$ & $F$):** Regresses squared standardized residuals $e_i^2 / \hat{\sigma}^2$ on regressors $Z$.
  $$\text{LM} = \frac{1}{2} \text{ESS}_{\text{aux}} \sim \chi^2(p)$$
- **White's General Test:** Tests for arbitrary non-linear heteroscedasticity by regressing $e_i^2$ on regressors, squares, and cross-products.
  $$e_i^2 = \alpha_0 + \sum \alpha_j X_j + \sum \alpha_{jk} X_j X_k + v_i$$
- **Goldfeld-Quandt Test:** Parametric variance ratio comparing partitioned sub-samples ordered by the suspected variance driver:
  $$F = \frac{\text{SSE}_2 / (n_2 - k)}{\text{SSE}_1 / (n_1 - k)} \sim F(n_2-k, n_1-k)$$

### 2. Serial Correlation Tests
- **Durbin-Watson Test ($d$):** First-order serial correlation test:
  $$d = \frac{\sum_{t=2}^T (e_t - e_{t-1})^2}{\sum_{t=1}^T e_t^2} \approx 2(1 - \hat{\rho})$$
- **Breusch-Godfrey LM Test ($AR(p)$):** Detects higher-order serial correlation even in models with lagged dependent variables:
  $$e_t = \mathbf{X}_t \boldsymbol{\gamma} + \sum_{j=1}^p \rho_j e_{t-j} + u_t$$
- **Ljung-Box Portmanteau Test:** Cumulative autocorrelation across multiple lags:
  $$Q = T(T+2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(m)$$

---

## 🛡️ Econometric Remedy Matrix

| Violation | Remedy Technique | Formula / Mechanism | Property |
| :--- | :--- | :--- | :--- |
| **Heteroscedasticity** | **White HC0** | $\widehat{\text{Var}}(\hat{\beta}) = (X'X)^{-1} X' \text{diag}(e_i^2) X (X'X)^{-1}$ | Consistent asymptotically |
| **Heteroscedasticity** | **MacKinnon-White HC3** | Weights by $(1 - h_{ii})^2$ leverage factors | **Recommended** for $N < 250$ |
| **Heteroscedasticity** | **Feasible GLS (FGLS)** | Regress $\ln(e_i^2) \sim X$, weight $w_i = 1 / \hat{\sigma}_i^2$ | Restores efficiency (BLUE) |
| **Serial Correlation** | **Prais-Winsten** | $y_t^* = y_t - \hat{\rho} y_{t-1}$, preserves $t=1$ with $\sqrt{1-\hat{\rho}^2}$ | Optimal for $AR(1)$ errors |
| **Both Present** | **Newey-West HAC** | Bartlett kernel lag weighting across $L$ lags | Robust to both simultaneously |

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/Vaibhav007-star/Non-constant-Variance-and-Serial-Correlation-Diagnostics-and-Remedies.git
cd Non-constant-Variance-and-Serial-Correlation-Diagnostics-and-Remedies

pip install -r requirements.txt
```

### 2. Launch the Interactive Web Platform
```bash
# Launch modern high-performance web platform (FastAPI + Interactive Plotly Cockpit)
python -m uvicorn app.server:app --port 8000 --reload

# Or on Windows, double-click:
run_app.bat

# Legacy Streamlit alternative:
streamlit run app/app.py
# or double-click run_streamlit.bat
```

### 3. Run Programmatic Audit via Python
```python
import pandas as pd
from src.pipeline.econometric_engine import EconometricAuditEngine

# Load dataset
df = pd.read_csv("data/macroeconomic_ts.csv")

# Initialize and audit
engine = EconometricAuditEngine(
    data=df,
    target_col="GDP_Growth",
    feature_cols=["Interest_Rate", "Money_Growth", "Unemployment_Rate"]
)
engine.fit_baseline()
diagnostics = engine.run_diagnostics()
remedies = engine.run_remedies()

# Display comparative scorecard
scorecard = engine.generate_scorecard()
print(scorecard)

# Print executive policy finding
print(engine.generate_audit_summary()["verdict"])
```

---

## 👤 Author
**Vaibhav**
- GitHub: [@Vaibhav007-star](https://github.com/Vaibhav007-star)
- Specialization: AI & Data Science Engineering • Econometric Modeling • Multi-Agent Systems
