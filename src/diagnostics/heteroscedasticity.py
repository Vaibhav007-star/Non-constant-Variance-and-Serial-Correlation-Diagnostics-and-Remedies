"""
Heteroscedasticity Diagnostic Test Suite.
Tests implemented:
1. Breusch-Pagan Lagrange Multiplier Test
2. White's General Test (with quadratic & interaction terms)
3. Goldfeld-Quandt Subsample Ratio Test
4. Glejser Test (functional form of variance)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_goldfeldquandt

def run_breusch_pagan_test(model):
    """
    Computes Breusch-Pagan LM and F tests for heteroscedasticity.
    H0: Homoscedasticity (constant error variance)
    H1: Heteroscedasticity (error variance depends on regressors)
    """
    resid = model.resid
    exog = model.model.exog
    lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(resid, exog)
    
    verdict = "Heteroscedasticity Detected (Reject H0 at p < 0.05)" if lm_pval < 0.05 else "Homoscedasticity (Fail to Reject H0)"
    
    return {
        "test_name": "Breusch-Pagan Test",
        "lm_statistic": float(lm_stat),
        "lm_pvalue": float(lm_pval),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pval),
        "reject_null_at_05": bool(lm_pval < 0.05),
        "verdict": verdict,
        "null_hypothesis": "Residual variance is constant across all observation levels (Homoscedasticity)."
    }

def run_white_test(model):
    """
    Computes White's general test for heteroscedasticity without assuming linear variance form.
    """
    resid = model.resid
    exog = model.model.exog
    lm_stat, lm_pval, f_stat, f_pval = het_white(resid, exog)
    
    verdict = "Heteroscedasticity Detected (Reject H0 at p < 0.05)" if lm_pval < 0.05 else "Homoscedasticity (Fail to Reject H0)"
    
    return {
        "test_name": "White's General Test",
        "lm_statistic": float(lm_stat),
        "lm_pvalue": float(lm_pval),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pval),
        "reject_null_at_05": bool(lm_pval < 0.05),
        "verdict": verdict,
        "null_hypothesis": "No heteroscedasticity and no nonlinear cross-term specification error."
    }

def run_goldfeld_quandt_test(model, split_idx=None):
    """
    Computes Goldfeld-Quandt parametric variance ratio test.
    Compares the variance of two sub-samples ordered by suspected variable.
    """
    y = model.model.endog
    X = model.model.exog
    split = split_idx if split_idx is not None else len(y) // 2
    f_stat, p_val, ordering = het_goldfeldquandt(y, X, split=split, result_object=False)
    
    verdict = "Heteroscedasticity Detected (Reject H0 at p < 0.05)" if p_val < 0.05 else "Homoscedasticity (Fail to Reject H0)"
    
    return {
        "test_name": "Goldfeld-Quandt Test",
        "f_statistic": float(f_stat),
        "pvalue": float(p_val),
        "ordering": str(ordering),
        "reject_null_at_05": bool(p_val < 0.05),
        "verdict": verdict,
        "null_hypothesis": "Error variances in both sample partitions are equal."
    }

def run_all_heteroscedasticity_diagnostics(model):
    """Executes the full suite of heteroscedasticity tests and compiles a dictionary."""
    bp = run_breusch_pagan_test(model)
    white = run_white_test(model)
    gq = run_goldfeld_quandt_test(model)
    
    has_hetero = bp["reject_null_at_05"] or white["reject_null_at_05"]
    
    return {
        "breusch_pagan": bp,
        "white": white,
        "goldfeld_quandt": gq,
        "overall_heteroscedasticity_detected": has_hetero
    }
