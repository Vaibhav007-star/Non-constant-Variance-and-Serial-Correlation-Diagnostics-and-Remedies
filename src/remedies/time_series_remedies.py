"""
Time-Series Remedies for Serial Correlation.
Implements:
1. Cochrane-Orcutt Iterative Procedure
2. Prais-Winsten AR(1) Estimation
3. First-Differencing
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

def cochrane_orcutt(y, X, max_iter=30, tol=1e-5):
    """
    Cochrane-Orcutt iterative method:
    Transforms variables: y_t* = y_t - rho * y_{t-1}
                          X_t* = X_t - rho * X_{t-1}
    Iterates until rho converges.
    Note: First observation is dropped.
    """
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    T = len(y_arr)
    
    # Initial OLS
    beta = sm.OLS(y_arr, X_arr).fit().params
    rho = 0.0
    
    for iteration in range(max_iter):
        resid = y_arr - X_arr @ beta
        # Regress e_t on e_{t-1} without intercept
        e_t = resid[1:]
        e_tm1 = resid[:-1]
        rho_new = float(np.sum(e_t * e_tm1) / np.sum(e_tm1 ** 2))
        
        # Check convergence
        if abs(rho_new - rho) < tol:
            rho = rho_new
            break
        rho = rho_new
        
        # Transform data
        y_star = y_arr[1:] - rho * y_arr[:-1]
        X_star = X_arr[1:] - rho * X_arr[:-1]
        
        model_star = sm.OLS(y_star, X_star).fit()
        beta = model_star.params
        
    y_star = y_arr[1:] - rho * y_arr[:-1]
    X_star = X_arr[1:] - rho * X_arr[:-1]
    final_model = sm.OLS(y_star, X_star).fit()
    
    return {
        "final_model": final_model,
        "rho": float(rho),
        "iterations": iteration + 1,
        "method": "Cochrane-Orcutt (Iterative)"
    }

def prais_winsten(y, X, max_iter=30, tol=1e-5):
    """
    Prais-Winsten procedure:
    Like Cochrane-Orcutt, but preserves the first observation:
    y_1* = sqrt(1 - rho^2) * y_1
    X_1* = sqrt(1 - rho^2) * X_1
    """
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    T = len(y_arr)
    
    beta = sm.OLS(y_arr, X_arr).fit().params
    rho = 0.0
    
    for iteration in range(max_iter):
        resid = y_arr - X_arr @ beta
        e_t = resid[1:]
        e_tm1 = resid[:-1]
        rho_new = float(np.sum(e_t * e_tm1) / np.sum(e_tm1 ** 2))
        # Restrict rho to (-0.999, 0.999)
        rho_new = np.clip(rho_new, -0.999, 0.999)
        
        if abs(rho_new - rho) < tol:
            rho = rho_new
            break
        rho = rho_new
        
        # Prais-Winsten transformation
        y_star = np.empty_like(y_arr)
        X_star = np.empty_like(X_arr)
        
        weight_first = np.sqrt(1 - rho ** 2)
        y_star[0] = weight_first * y_arr[0]
        X_star[0] = weight_first * X_arr[0]
        
        y_star[1:] = y_arr[1:] - rho * y_arr[:-1]
        X_star[1:] = X_arr[1:] - rho * X_arr[:-1]
        
        model_star = sm.OLS(y_star, X_star).fit()
        beta = model_star.params
        
    y_star = np.empty_like(y_arr)
    X_star = np.empty_like(X_arr)
    weight_first = np.sqrt(1 - rho ** 2)
    y_star[0] = weight_first * y_arr[0]
    X_star[0] = weight_first * X_arr[0]
    y_star[1:] = y_arr[1:] - rho * y_arr[:-1]
    X_star[1:] = X_arr[1:] - rho * X_arr[:-1]
    final_model = sm.OLS(y_star, X_star).fit()
    
    return {
        "final_model": final_model,
        "rho": float(rho),
        "iterations": iteration + 1,
        "method": "Prais-Winsten (Full Sample Preserved)"
    }
