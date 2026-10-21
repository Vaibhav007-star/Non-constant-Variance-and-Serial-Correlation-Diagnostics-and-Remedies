"""
Unit tests for econometric remedies (HC3, HAC, Prais-Winsten, FGLS).
"""

import pytest
import numpy as np
import statsmodels.api as sm

from src.remedies.robust_standard_errors import fit_ols_with_robust_se
from src.remedies.gls_wls import fit_feasible_gls
from src.remedies.time_series_remedies import prais_winsten, cochrane_orcutt

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
