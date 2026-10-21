"""
Serial Correlation (Autocorrelation) Diagnostic Test Suite.
Tests implemented:
1. Durbin-Watson Test (First-order AR(1))
2. Breusch-Godfrey Lagrange Multiplier Test (Higher-order AR(p))
3. Ljung-Box Portmanteau Q-Test (Cumulative serial correlation)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox

def run_durbin_watson_test(model):
    """
    Computes Durbin-Watson test statistic:
    d = sum((e_t - e_{t-1})^2) / sum(e_t^2)
    Rules of thumb:
    d ~ 2: No serial correlation
    d < 1.5: Positive serial correlation
    d > 2.5: Negative serial correlation
    """
    resid = model.resid
    dw_stat = float(durbin_watson(resid))
    
    if dw_stat < 1.5:
        verdict = f"Positive Serial Correlation Detected (d={dw_stat:.3f} < 1.5)"
        has_ac = True
    elif dw_stat > 2.5:
        verdict = f"Negative Serial Correlation Detected (d={dw_stat:.3f} > 2.5)"
        has_ac = True
    else:
        verdict = f"No Strong First-Order Serial Correlation (d={dw_stat:.3f} in [1.5, 2.5])"
        has_ac = False
        
    return {
        "test_name": "Durbin-Watson Test",
        "dw_statistic": dw_stat,
        "verdict": verdict,
        "serial_correlation_detected": has_ac,
        "null_hypothesis": "No first-order autocorrelation in residuals (rho = 0, d = 2.0)."
    }

def run_breusch_godfrey_test(model, nlags=4):
    """
    Computes Breusch-Godfrey LM test for higher order serial correlation AR(p).
    """
    lm_stat, lm_pval, f_stat, f_pval = acorr_breusch_godfrey(model, nlags=nlags)
    
    verdict = f"Serial Correlation up to lag {nlags} Detected (p < 0.05)" if lm_pval < 0.05 else f"No Serial Correlation up to lag {nlags} (Fail to Reject H0)"
    
    return {
        "test_name": f"Breusch-Godfrey LM Test (lags={nlags})",
        "lm_statistic": float(lm_stat),
        "lm_pvalue": float(lm_pval),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pval),
        "reject_null_at_05": bool(lm_pval < 0.05),
        "verdict": verdict,
        "null_hypothesis": f"No serial correlation up to order {nlags} in disturbances."
    }

def run_ljung_box_test(model, lags=8):
    """
    Computes Ljung-Box Q-statistic across multiple lags.
    """
    resid = model.resid
    lb_df = acorr_ljungbox(resid, lags=lags, return_df=True)
    
    min_pval = lb_df["lb_pvalue"].min()
    has_ac = (min_pval < 0.05)
    
    return {
        "test_name": "Ljung-Box Portmanteau Test",
        "table": lb_df.to_dict(orient="index"),
        "min_pvalue": float(min_pval),
        "reject_null_at_05": bool(has_ac),
        "verdict": "Cumulative Autocorrelation Detected (p < 0.05)" if has_ac else "No Significant Autocorrelation Detected across lags."
    }

def run_all_serial_correlation_diagnostics(model, nlags=4):
    dw = run_durbin_watson_test(model)
    bg = run_breusch_godfrey_test(model, nlags=nlags)
    lb = run_ljung_box_test(model, lags=min(10, len(model.resid)//5))
    
    has_sc = dw["serial_correlation_detected"] or bg["reject_null_at_05"]
    
    return {
        "durbin_watson": dw,
        "breusch_godfrey": bg,
        "ljung_box": lb,
        "overall_serial_correlation_detected": has_sc
    }
