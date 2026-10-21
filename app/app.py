"""
Interactive Econometric Diagnostics & Remedies Web Application.
Built with Streamlit.
Author: Vaibhav (@Vaibhav007-star)
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.econometric_engine import EconometricAuditEngine
from src.visualizations.diagnostic_plots import (
    plot_residuals_vs_fitted,
    plot_scale_location,
    plot_qq,
    plot_autocorrelation,
    plot_lag_scatter
)

st.set_page_config(
    page_title="Econometric Diagnostics & Remedies Cockpit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------- HEADER -----------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <h1 class="main-title">📈 Econometric Diagnostics & Remedies Platform</h1>
            <p class="main-subtitle">
                Advanced statistical audit suite for <b>Non-constant Variance (Heteroscedasticity)</b> and <b>Serial Correlation (Autocorrelation)</b> in Linear Regression Models.
            </p>
        </div>
        <div style="text-align: right;">
            <span class="status-badge badge-success">● Active Engine</span><br>
            <span style="font-size: 11px; color: #94A3B8;">OLS • HC3 • HAC • FGLS • Prais-Winsten</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### ⚙️ Dataset & Model Setup")

dataset_choice = st.sidebar.selectbox(
    "Choose Econometric Dataset",
    [
        "Macroeconomic Time-Series (Serial Correlation & ARCH)",
        "Housing Prices Cross-Section (Heteroscedasticity)",
        "Corporate Wage & Human Capital (Variance Divergence)",
        "Upload Custom CSV..."
    ]
)

data = None
data_dir = os.path.join(PROJECT_ROOT, "data")

if dataset_choice == "Macroeconomic Time-Series (Serial Correlation & ARCH)":
    csv_file = os.path.join(data_dir, "macroeconomic_ts.csv")
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        default_target = "GDP_Growth"
        default_features = ["Interest_Rate", "Money_Growth", "Unemployment_Rate"]
elif dataset_choice == "Housing Prices Cross-Section (Heteroscedasticity)":
    csv_file = os.path.join(data_dir, "housing_prices.csv")
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        default_target = "Price"
        default_features = ["Square_Feet", "Lot_Size", "Bedrooms", "Age_Years", "School_Rating"]
elif dataset_choice == "Corporate Wage & Human Capital (Variance Divergence)":
    csv_file = os.path.join(data_dir, "corporate_wages.csv")
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        default_target = "Hourly_Wage"
        default_features = ["Education_Years", "Experience_Years", "Tenure_Months", "Firm_Size"]
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        default_target = num_cols[0] if num_cols else ""
        default_features = num_cols[1:] if len(num_cols) > 1 else []

if data is None:
    st.error("No dataset loaded. Please check that dataset files exist in `data/` or upload a CSV.")
    st.stop()

# Target and Feature Selection
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("Dataset contains no numeric variables for regression modeling.")
    st.stop()

target_col = st.sidebar.selectbox(
    "Dependent Variable (Y)",
    numeric_cols,
    index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
)

available_features = [c for c in numeric_cols if c != target_col]
feature_cols = st.sidebar.multiselect(
    "Independent Regressors (X)",
    available_features,
    default=[f for f in default_features if f in available_features] or available_features[:3]
)

add_constant = st.sidebar.checkbox("Include Constant / Intercept (beta_0)", value=True)
sig_level = st.sidebar.slider("Significance Level (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

if not feature_cols:
    st.warning("Please select at least one independent regressor (X) in the sidebar.")
    st.stop()

# ----------------- EXECUTE AUDIT ENGINE -----------------
engine = EconometricAuditEngine(
    data=data,
    target_col=target_col,
    feature_cols=feature_cols,
    add_constant=add_constant
)

ols_model = engine.fit_baseline()
diag_results = engine.run_diagnostics()
remedy_models = engine.run_remedies()
scorecard_df = engine.generate_scorecard()
audit_summary = engine.generate_audit_summary()

# ----------------- TOP METRIC BAR -----------------
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Observations (N)", f"{int(ols_model.nobs):,}")
with m2:
    st.metric("R² Score", f"{ols_model.rsquared:.4f}")
with m3:
    st.metric("Adj. R²", f"{ols_model.rsquared_adj:.4f}")
with m4:
    st.metric("Residual Std Err", f"{np.sqrt(ols_model.scale):.3f}")
with m5:
    st.metric("F-Statistic", f"{ols_model.fvalue:.2f}")
with m6:
    dw_val = diag_results["serial_correlation"]["durbin_watson"]["dw_statistic"]
    st.metric("Durbin-Watson", f"{dw_val:.3f}")

# ----------------- STATUS BANNER -----------------
has_het = diag_results["heteroscedasticity"]["overall_heteroscedasticity_detected"]
has_ser = diag_results["serial_correlation"]["overall_serial_correlation_detected"]

c_banner1, c_banner2 = st.columns(2)
with c_banner1:
    if has_het:
        st.markdown(
            '<div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239,68,68,0.4); padding: 12px 16px; border-radius: 8px;">'
            '<span style="color: #EF4444; font-weight: 700;">⚠️ Heteroscedasticity Detected!</span><br>'
            '<span style="font-size: 13px; color: #CBD5E1;">Non-constant residual variance verified by Breusch-Pagan / White tests. Classical SEs are biased.</span>'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16,185,129,0.4); padding: 12px 16px; border-radius: 8px;">'
            '<span style="color: #10B981; font-weight: 700;">✅ Homoscedasticity Satisfied</span><br>'
            '<span style="font-size: 13px; color: #CBD5E1;">Constant variance assumption holds. OLS standard errors are unbiased.</span>'
            '</div>', unsafe_allow_html=True
        )

with c_banner2:
    if has_ser:
        st.markdown(
            '<div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245,158,11,0.4); padding: 12px 16px; border-radius: 8px;">'
            '<span style="color: #F59E0B; font-weight: 700;">⚠️ Serial Correlation Detected!</span><br>'
            '<span style="font-size: 13px; color: #CBD5E1;">Residual persistence verified by Durbin-Watson / Breusch-Godfrey tests. Standard errors are underestimated.</span>'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16,185,129,0.4); padding: 12px 16px; border-radius: 8px;">'
            '<span style="color: #10B981; font-weight: 700;">✅ Error Independence Satisfied</span><br>'
            '<span style="font-size: 13px; color: #CBD5E1;">No significant autocorrelation detected in regression disturbances.</span>'
            '</div>', unsafe_allow_html=True
        )

st.write("")

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Baseline OLS Model",
    "🔍 Visual Diagnostics Gallery",
    "🧪 Statistical Hypothesis Tests",
    "🛡️ Remedies & Robust Scorecard",
    "📋 Executive Audit Report"
])

# ----- TAB 1: BASELINE OLS -----
with tab1:
    st.subheader("Baseline Ordinary Least Squares (OLS) Regression")
    
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        # Format parameter table
        params_df = pd.DataFrame({
            "Coefficient (beta)": ols_model.params,
            "Std. Error (SE)": ols_model.bse,
            "t-Statistic": ols_model.tvalues,
            "p-Value": ols_model.pvalues,
            "95% CI Lower": ols_model.conf_int()[0],
            "95% CI Upper": ols_model.conf_int()[1]
        })
        st.dataframe(
            params_df.style.format({
                "Coefficient (beta)": "{:.4f}",
                "Std. Error (SE)": "{:.4f}",
                "t-Statistic": "{:.3f}",
                "p-Value": "{:.4e}",
                "95% CI Lower": "{:.4f}",
                "95% CI Upper": "{:.4f}"
            }).background_gradient(subset=["p-Value"], cmap="Reds_r"),
            use_container_width=True
        )
        
    with col_t2:
        st.markdown("#### Model Specification Statistics")
        stats_data = {
            "Metric": ["Dependent Variable", "No. Observations", "Df Residuals", "R-squared", "Adj. R-squared", "F-Statistic", "Prob (F-statistic)", "Log-Likelihood", "AIC", "BIC"],
            "Value": [
                target_col,
                f"{int(ols_model.nobs)}",
                f"{int(ols_model.df_resid)}",
                f"{ols_model.rsquared:.4f}",
                f"{ols_model.rsquared_adj:.4f}",
                f"{ols_model.fvalue:.3f}",
                f"{ols_model.f_pvalue:.4e}",
                f"{ols_model.llf:.2f}",
                f"{ols_model.aic:.2f}",
                f"{ols_model.bic:.2f}"
            ]
        }
        st.table(pd.DataFrame(stats_data))

# ----- TAB 2: VISUAL DIAGNOSTICS -----
with tab2:
    st.subheader("Visual Residual Diagnostics")
    st.markdown("Visual inspection allows immediate identification of non-constant variance shapes (funnels/diamonds) and temporal dependence clusters.")
    
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.markdown("#### 1. Residuals vs. Fitted Values")
        fig1 = plot_residuals_vs_fitted(ols_model)
        st.pyplot(fig1)
        plt.close(fig1)
        st.caption("A broadening funnel pattern indicates heteroscedasticity (error variance expands with Y_hat).")

    with vcol2:
        st.markdown("#### 2. Scale-Location Plot")
        fig2 = plot_scale_location(ols_model)
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption("An upward-trending LOWESS line confirms that error dispersion increases across fitted values.")

    vcol3, vcol4 = st.columns(2)
    with vcol3:
        st.markdown("#### 3. Autocorrelation Function (ACF)")
        fig3 = plot_autocorrelation(ols_model)
        st.pyplot(fig3)
        plt.close(fig3)
        st.caption("Bars exceeding the light blue 95% Bartlett confidence band demonstrate statistically significant serial correlation.")

    with vcol4:
        st.markdown("#### 4. Residual Lag Scatter Plot (e_t vs e_{t-1})")
        fig4 = plot_lag_scatter(ols_model)
        st.pyplot(fig4)
        plt.close(fig4)
        st.caption("A diagonal cloud indicates positive first-order autocorrelation AR(1) where positive errors follow positive errors.")

# ----- TAB 3: STATISTICAL TESTS -----
with tab3:
    st.subheader("Formal Econometric Hypothesis Testing")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("### 🔍 Heteroscedasticity Tests")
        
        # Breusch-Pagan
        bp = diag_results["heteroscedasticity"]["breusch_pagan"]
        st.markdown(f"""
        **1. Breusch-Pagan Lagrange Multiplier Test**
        - **LM Statistic:** `{bp['lm_statistic']:.4f}` (p-value: `{bp['lm_pvalue']:.4e}`)
        - **F Statistic:** `{bp['f_statistic']:.4f}` (p-value: `{bp['f_pvalue']:.4e}`)
        - **Verdict:** {'🚨 **' + bp['verdict'] + '**' if bp['reject_null_at_05'] else '✅ **' + bp['verdict'] + '**'}
        """)
        
        # White Test
        white = diag_results["heteroscedasticity"]["white"]
        st.markdown(f"""
        **2. White's General Test (with Cross-Terms)**
        - **LM Statistic:** `{white['lm_statistic']:.4f}` (p-value: `{white['lm_pvalue']:.4e}`)
        - **F Statistic:** `{white['f_statistic']:.4f}` (p-value: `{white['f_pvalue']:.4e}`)
        - **Verdict:** {'🚨 **' + white['verdict'] + '**' if white['reject_null_at_05'] else '✅ **' + white['verdict'] + '**'}
        """)
        
        # Goldfeld-Quandt
        gq = diag_results["heteroscedasticity"]["goldfeld_quandt"]
        st.markdown(f"""
        **3. Goldfeld-Quandt Variance Ratio Test**
        - **F Statistic:** `{gq['f_statistic']:.4f}` (p-value: `{gq['pvalue']:.4e}`)
        - **Verdict:** {'🚨 **' + gq['verdict'] + '**' if gq['reject_null_at_05'] else '✅ **' + gq['verdict'] + '**'}
        """)

    with t_col2:
        st.markdown("### 🔄 Serial Correlation Tests")
        
        # Durbin-Watson
        dw = diag_results["serial_correlation"]["durbin_watson"]
        st.markdown(f"""
        **1. Durbin-Watson Test (First-Order AR(1))**
        - **Test Statistic (d):** `{dw['dw_statistic']:.4f}`
        - **Benchmark:** `d ≈ 2` (No Autocorrelation), `d < 1.5` (Positive), `d > 2.5` (Negative)
        - **Verdict:** {'🚨 **' + dw['verdict'] + '**' if dw['serial_correlation_detected'] else '✅ **' + dw['verdict'] + '**'}
        """)
        
        # Breusch-Godfrey
        bg = diag_results["serial_correlation"]["breusch_godfrey"]
        st.markdown(f"""
        **2. Breusch-Godfrey LM Test (Higher Order AR(p))**
        - **LM Statistic:** `{bg['lm_statistic']:.4f}` (p-value: `{bg['lm_pvalue']:.4e}`)
        - **F Statistic:** `{bg['f_statistic']:.4f}` (p-value: `{bg['f_pvalue']:.4e}`)
        - **Verdict:** {'🚨 **' + bg['verdict'] + '**' if bg['reject_null_at_05'] else '✅ **' + bg['verdict'] + '**'}
        """)

        # Ljung-Box
        lb = diag_results["serial_correlation"]["ljung_box"]
        st.markdown(f"""
        **3. Ljung-Box Portmanteau Test (Cumulative Lags)**
        - **Minimum p-value:** `{lb['min_pvalue']:.4e}`
        - **Verdict:** {'🚨 **' + lb['verdict'] + '**' if lb['reject_null_at_05'] else '✅ **' + lb['verdict'] + '**'}
        """)

# ----- TAB 4: REMEDIES & SCORECARD -----
with tab4:
    st.subheader("Model Remedies & Standard Error Scorecard")
    st.markdown("""
    When Gauss-Markov assumptions are violated:
    - **OLS Coefficients** remain **unbiased** and **consistent**.
    - **OLS Standard Errors** are **biased** (usually downward), causing artificially inflated t-statistics and false positives.
    - **Remedy 1 (White HC3):** Adjusts standard errors for non-constant variance without changing point estimates.
    - **Remedy 2 (Newey-West HAC):** Simultaneously corrects standard errors for both heteroscedasticity AND serial correlation.
    - **Remedy 3 (Feasible GLS):** Re-weights observations by inverse estimated error variance for efficient point estimates.
    """)
    
    st.markdown("#### Side-by-Side Comparison Scorecard")
    st.dataframe(
        scorecard_df.style.format({
            "OLS_Coeff": "{:.4f}",
            "OLS_SE": "{:.4f}",
            "OLS_pVal": "{:.4e}",
            "HC3_SE": "{:.4f}",
            "HC3_pVal": "{:.4e}",
            "HAC_SE": "{:.4f}",
            "HAC_pVal": "{:.4e}",
            "FGLS_Coeff": "{:.4f}",
            "FGLS_SE": "{:.4f}",
            "FGLS_pVal": "{:.4e}"
        }),
        use_container_width=True
    )
    
    # Compute SE inflation ratios
    scorecard_df["SE_Ratio_HC3_vs_OLS"] = scorecard_df["HC3_SE"] / scorecard_df["OLS_SE"]
    scorecard_df["SE_Ratio_HAC_vs_OLS"] = scorecard_df["HAC_SE"] / scorecard_df["OLS_SE"]
    
    st.markdown("#### Standard Error Adjustment Factors (SE_Robust / SE_OLS)")
    ratio_df = scorecard_df[["Variable", "SE_Ratio_HC3_vs_OLS", "SE_Ratio_HAC_vs_OLS"]]
    st.dataframe(
        ratio_df.style.format({
            "SE_Ratio_HC3_vs_OLS": "{:.2f}x",
            "SE_Ratio_HAC_vs_OLS": "{:.2f}x"
        }).background_gradient(subset=["SE_Ratio_HAC_vs_OLS"], cmap="Blues"),
        use_container_width=True
    )

# ----- TAB 5: AUDIT REPORT & EXPORT -----
with tab5:
    st.subheader("Executive Econometric Audit Summary")
    
    st.markdown(f"### Overall Finding: **{audit_summary['verdict']}**")
    
    st.markdown("#### Recommended Econometric Policy:")
    for rec in audit_summary["recommendations"]:
        st.markdown(f"- 📌 {rec}")
        
    st.markdown("---")
    
    # Generate Markdown export
    report_md = f"""# Econometric Audit Report: Diagnostics & Remedies
**Generated by**: Econometric Diagnostics & Remedies Cockpit
**Target Variable**: `{target_col}`
**Regressors**: `{', '.join(feature_cols)}`
**Sample Size (N)**: `{int(ols_model.nobs)}`

---

## 1. Baseline Model Performance
- **R-squared**: `{ols_model.rsquared:.4f}` (Adjusted R²: `{ols_model.rsquared_adj:.4f}`)
- **F-statistic**: `{ols_model.fvalue:.3f}` (p = `{ols_model.f_pvalue:.4e}`)
- **Residual Standard Error**: `{np.sqrt(ols_model.scale):.4f}`
- **Durbin-Watson Statistic**: `{diag_results['serial_correlation']['durbin_watson']['dw_statistic']:.4f}`

## 2. Diagnostic Hypothesis Tests
- **Breusch-Pagan Test**: LM = `{diag_results['heteroscedasticity']['breusch_pagan']['lm_statistic']:.4f}`, p = `{diag_results['heteroscedasticity']['breusch_pagan']['lm_pvalue']:.4e}` ({diag_results['heteroscedasticity']['breusch_pagan']['verdict']})
- **White's Test**: LM = `{diag_results['heteroscedasticity']['white']['lm_statistic']:.4f}`, p = `{diag_results['heteroscedasticity']['white']['lm_pvalue']:.4e}` ({diag_results['heteroscedasticity']['white']['verdict']})
- **Durbin-Watson**: d = `{diag_results['serial_correlation']['durbin_watson']['dw_statistic']:.4f}` ({diag_results['serial_correlation']['durbin_watson']['verdict']})
- **Breusch-Godfrey LM**: LM = `{diag_results['serial_correlation']['breusch_godfrey']['lm_statistic']:.4f}`, p = `{diag_results['serial_correlation']['breusch_godfrey']['lm_pvalue']:.4e}` ({diag_results['serial_correlation']['breusch_godfrey']['verdict']})

## 3. Executive Policy Recommendation
{chr(10).join([f"- {r}" for r in audit_summary['recommendations']])}
"""
    st.download_button(
        label="📥 Download Full Econometric Audit Report (Markdown)",
        data=report_md,
        file_name="econometric_audit_report.md",
        mime="text/markdown"
    )
