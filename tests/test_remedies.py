"""
Unit tests for econometric remedies (HC3, HAC, Prais-Winsten, FGLS).
"""

import pytest
import numpy as np
import statsmodels.api as sm

from src.remedies.robust_standard_errors import fit_ols_with_robust_se, compare_robust_standard_errors
from src.remedies.gls_wls import fit_feasible_gls, fit_wls
from src.remedies.time_series_remedies import prais_winsten, cochrane_orcutt
from src.remedies.transformations import apply_log_transform, apply_box_cox_transform

def test_robust_standard_errors_run():
    np.random.seed(789)
    N = 150
    x = np.random.uniform(1, 10, N)
    y = 1.5 + 2.0 * x + np.random.normal(0, x)
    X = sm.add_constant(x)
    
    ols_raw = sm.OLS(y, X).fit()
    hc3 = fit_ols_with_robust_se(y, X, cov_type="HC3")
    hac = fit_ols_with_robust_se(y, X, cov_type="HAC")
    
    # Coefficients remain identical
    np.testing.assert_allclose(ols_raw.params, hc3.params)
    np.testing.assert_allclose(ols_raw.params, hac.params)
    
    # Standard errors are adjusted
    assert len(hc3.bse) == 2
    assert len(hac.bse) == 2

def test_compare_robust_standard_errors():
    np.random.seed(789)
    N = 100
    x = np.random.uniform(1, 5, N)
    y = 2.0 + 1.5 * x + np.random.normal(0, 0.5 * x)
    X = sm.add_constant(x)
    df_comp = compare_robust_standard_errors(y, X)
    assert "SE_HC3" in df_comp.columns
    assert "SE_NeweyWest_HAC" in df_comp.columns
    assert len(df_comp) == 2

def test_feasible_gls():
    np.random.seed(101)
    N = 200
    x = np.random.uniform(2, 20, N)
    y = 5.0 + 1.2 * x + np.random.normal(0, 0.4 * x)
    X = sm.add_constant(x)
    
    fgls, weights, _ = fit_feasible_gls(y, X)
    assert fgls is not None
    assert len(weights) == N
    assert np.all(weights > 0)

def test_fit_wls():
    np.random.seed(101)
    N = 50
    x = np.random.uniform(1, 10, N)
    y = 3.0 + 2.0 * x + np.random.normal(0, 1, N)
    X = sm.add_constant(x)
    weights = np.ones(N) * 2.0
    wls_model = fit_wls(y, X, weights=weights)
    assert wls_model is not None
    assert len(wls_model.params) == 2

def test_prais_winsten():
    np.random.seed(202)
    T = 100
    x = np.linspace(0, 10, T)
    e = np.zeros(T)
    for t in range(1, T):
        e[t] = 0.6 * e[t-1] + np.random.normal(0, 1)
    y = 2.0 + 1.5 * x + e
    X = sm.add_constant(x)
    
    pw = prais_winsten(y, X)
    assert pw["iterations"] >= 1
    assert -1.0 < pw["rho"] < 1.0

def test_cochrane_orcutt():
    np.random.seed(303)
    T = 100
    x = np.linspace(0, 10, T)
    e = np.zeros(T)
    for t in range(1, T):
        e[t] = 0.5 * e[t-1] + np.random.normal(0, 1)
    y = 3.0 + 1.8 * x + e
    X = sm.add_constant(x)
    
    co = cochrane_orcutt(y, X)
    assert co["iterations"] >= 1
    assert -1.0 < co["rho"] < 1.0

def test_transformations():
    y = np.array([10.0, 20.0, 50.0, 100.0, 200.0])
    y_log, offset = apply_log_transform(y)
    assert len(y_log) == 5
    assert offset == 0.0

    y_neg = np.array([-5.0, 0.0, 5.0, 10.0])
    y_log_neg, offset_neg = apply_log_transform(y_neg)
    assert offset_neg > 5.0
    assert np.all(np.isfinite(y_log_neg))

    y_bc, lam, bc_off = apply_box_cox_transform(y)
    assert len(y_bc) == 5
    assert isinstance(lam, float)
