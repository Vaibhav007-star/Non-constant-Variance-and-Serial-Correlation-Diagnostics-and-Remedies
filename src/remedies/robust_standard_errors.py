"""
Robust Standard Errors and Covariance Matrix Estimation.
Implements:
- White's HC0 (1980)
- HC1 (degrees of freedom adjusted)
- HC2 (leverage adjusted)
- HC3 (jackknife approximation - recommended for small/moderate samples)
- Newey-West HAC (Heteroscedasticity and Autocorrelation Consistent)
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np

def fit_ols_with_robust_se(y, X, cov_type="HC3", maxlags=None):
    """
    Fits OLS and adjusts the covariance matrix of parameters.
    cov_type options: 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC'
    """
    ols_model = sm.OLS(y, X).fit()
    
    if cov_type == "nonrobust":
        return ols_model
    elif cov_type in ["HC0", "HC1", "HC2", "HC3"]:
        return ols_model.get_robustcov_results(cov_type=cov_type)
    elif cov_type == "HAC":
        lags = maxlags if maxlags is not None else int(np.floor(4 * ((len(y) / 100) ** (2/9))))
        return ols_model.get_robustcov_results(cov_type="HAC", maxlags=lags)
    else:
        raise ValueError(f"Unsupported cov_type: {cov_type}")

def compare_robust_standard_errors(y, X):
    """
    Generates a comparison table of Standard Errors across:
    - OLS Baseline (unadjusted)
    - White HC0
    - White HC1
    - MacKinnon-White HC3
    - Newey-West HAC
    """
    cov_types = ["nonrobust", "HC0", "HC1", "HC3", "HAC"]
    results = {}
    
    for c in cov_types:
        model = fit_ols_with_robust_se(y, X, cov_type=c)
        results[c] = {
            "params": model.params,
            "bse": model.bse,
            "tvalues": model.tvalues,
            "pvalues": model.pvalues,
            "conf_lower": model.conf_int()[0],
            "conf_upper": model.conf_int()[1]
        }
        
    df_comparison = pd.DataFrame({
        "OLS_Coeff": results["nonrobust"]["params"],
        "SE_Standard": results["nonrobust"]["bse"],
        "SE_HC0": results["HC0"]["bse"],
        "SE_HC1": results["HC1"]["bse"],
        "SE_HC3": results["HC3"]["bse"],
        "SE_NeweyWest_HAC": results["HAC"]["bse"],
        "Pval_Standard": results["nonrobust"]["pvalues"],
        "Pval_HC3": results["HC3"]["pvalues"],
        "Pval_HAC": results["HAC"]["pvalues"]
    })
    
    return df_comparison
