"""
Generalized Least Squares (GLS) & Weighted Least Squares (WLS).
Remedies for heteroscedasticity by reweighting observations with inverse error variances.
"""

import statsmodels.api as sm
import numpy as np
import pandas as pd

def fit_wls(y, X, weights):
    """
    Weighted Least Squares (WLS).
    weights: array-like of weights w_i = 1 / sigma_i^2
    """
    wls_model = sm.WLS(y, X, weights=weights).fit()
    return wls_model

def fit_feasible_gls(y, X):
    """
    Feasible Generalized Least Squares (FGLS) for unknown heteroscedasticity:
    Step 1: Fit baseline OLS, obtain residuals e_i.
    Step 2: Regress log(e_i^2) on X (or log of X): ln(e_i^2) = z_i' alpha + v_i.
    Step 3: Compute fitted values g_i and estimated variances sigma_i^2 = exp(g_i).
    Step 4: Compute weights w_i = 1 / sigma_i^2.
    Step 5: Estimate model using WLS with weights w_i.
    """
    ols = sm.OLS(y, X).fit()
    log_resid2 = np.log(np.maximum(ols.resid ** 2, 1e-10))
    
    aux_model = sm.OLS(log_resid2, X).fit()
    fitted_log_var = aux_model.fittedvalues
    est_variance = np.exp(fitted_log_var)
    weights = 1.0 / est_variance
    
    fgls_model = sm.WLS(y, X, weights=weights).fit()
    return fgls_model, weights, aux_model
