"""
Publication-Quality Diagnostic Visualizations.
Implements:
1. Residuals vs. Fitted Values (Heteroscedasticity funnel detection)
2. Scale-Location Plot (sqrt(|std resid|) vs fitted)
3. Normal Q-Q Plot
4. Autocorrelation Function (ACF) and PACF plots
5. Residual Lag-1 Scatter Plot (e_t vs e_{t-1} for serial correlation)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf, pacf

# Set cohesive aesthetic style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def plot_residuals_vs_fitted(model, title="Residuals vs Fitted Values (Heteroscedasticity Diagnostic)"):
    """
    Plots residuals against fitted values.
    Funnel / cone shape reveals non-constant variance (heteroscedasticity).
    """
    fitted = model.fittedvalues
    resid = model.resid
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, resid, alpha=0.6, color="#2563EB", edgecolors="none", s=35)
    ax.axhline(0, color="#EF4444", linestyle="--", linewidth=1.5, label="Zero Error Line")
    
    # Lowess smooth curve to show trend in variance/mean
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(resid, fitted, frac=0.6)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="#DC2626", linewidth=2, label="LOWESS Fit")
    except Exception:
        pass
        
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Fitted Values ($\hat{Y}$)", fontsize=11)
    ax.set_ylabel("Residuals ($e_i$)", fontsize=11)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

def plot_scale_location(model, title="Scale-Location Plot (Homoscedasticity Check)"):
    """
    Plots sqrt(|standardized residuals|) vs fitted values.
    A horizontal line indicates homoscedasticity. Upward slope indicates increasing variance.
    """
    fitted = model.fittedvalues
    # Standardized residuals
    resid = model.resid
    std_resid = resid / np.std(resid, ddof=1)
    sqrt_abs_std_resid = np.sqrt(np.abs(std_resid))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, sqrt_abs_std_resid, alpha=0.6, color="#059669", edgecolors="none", s=35)
    
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(sqrt_abs_std_resid, fitted, frac=0.6)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="#B91C1C", linewidth=2, label="Trend (LOWESS)")
    except Exception:
        pass
        
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Fitted Values ($\hat{Y}$)", fontsize=11)
    ax.set_ylabel("$\sqrt{|Standardized Residuals|}$", fontsize=11)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

def plot_qq(model, title="Normal Q-Q Plot of Residuals"):
    """Normal Q-Q plot to verify normality of residuals."""
    resid = model.resid
    fig, ax = plt.subplots(figsize=(7, 5))
    qqplot(resid, line="45", fit=True, ax=ax)
    ax.get_lines()[0].set_color("#6366F1")
    ax.get_lines()[0].set_markersize(5)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color("#DC2626")
    ax.get_lines()[1].set_linewidth(1.8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig

def plot_autocorrelation(model, nlags=20, title="Autocorrelation Function (ACF) of Residuals"):
    """
    Plots the ACF of regression residuals with 95% Bartlett confidence bands.
    Significant spikes outside bands indicate serial correlation.
    """
    resid = model.resid
    nlags = min(nlags, len(resid) // 3)
    acf_vals, confint = acf(resid, nlags=nlags, alpha=0.05)
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lags = np.arange(len(acf_vals))
    ax.vlines(lags, [0], acf_vals, color="#3B82F6", linewidth=2.5)
    ax.scatter(lags, acf_vals, color="#1D4ED8", s=30, zorder=3)
    
    # 95% confidence intervals
    ci_lower = confint[:, 0] - acf_vals
    ci_upper = confint[:, 1] - acf_vals
    ax.fill_between(lags, ci_lower, ci_upper, color="#93C5FD", alpha=0.35, label="95% Confidence Band")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Lag (k)", fontsize=11)
    ax.set_ylabel("Autocorrelation $\rho_k$", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

def plot_lag_scatter(model, title="Residual Lag Plot ($e_t$ vs. $e_{t-1}$ — Serial Correlation)"):
    """
    Scatter plot of residual at time t against residual at time t-1.
    A clear linear pattern indicates strong first-order autocorrelation AR(1).
    """
    resid = np.asarray(model.resid)
    e_t = resid[1:]
    e_tm1 = resid[:-1]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(e_tm1, e_t, alpha=0.65, color="#D97706", edgecolors="none", s=40)
    
    # Linear fit line
    m, b = np.polyfit(e_tm1, e_t, 1)
    ax.plot(e_tm1, m * e_tm1 + b, color="#B45309", linewidth=2, label=f"Fit ($\hat{{\rho}} = {m:.3f}$)")
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Residual at $t-1$ ($e_{t-1}$)", fontsize=11)
    ax.set_ylabel("Residual at $t$ ($e_t$)", fontsize=11)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
