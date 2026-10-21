"""
Unit tests for heteroscedasticity and serial correlation diagnostics.
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.diagnostics.heteroscedasticity import (
    run_breusch_pagan_test,
    run_white_test,
    run_goldfeld_quandt_test
)
from src.diagnostics.serial_correlation import (
    run_durbin_watson_test,
    run_breusch_godfrey_test
)

@pytest.fixture
def heteroscedastic_model():
    np.random.seed(123)
    N = 300
    x = np.linspace(1, 10, N)
    # Variance increases with x^2
    sigma = 0.5 * (x ** 1.5)
    e = np.random.normal(0, sigma)
    y = 2.0 + 3.0 * x + e
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()

@pytest.fixture
def autocorrelated_model():
    np.random.seed(456)
    T = 250
    x = np.random.normal(0, 1, T)
    e = np.zeros(T)
    for t in range(1, T):
        e[t] = 0.75 * e[t-1] + np.random.normal(0, 1)
    y = 1.0 + 2.5 * x + e
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()

def test_breusch_pagan_detects_heteroscedasticity(heteroscedastic_model):
    res = run_breusch_pagan_test(heteroscedastic_model)
    assert res["lm_pvalue"] < 0.05
    assert res["reject_null_at_05"] is True

def test_white_detects_heteroscedasticity(heteroscedastic_model):
    res = run_white_test(heteroscedastic_model)
    assert res["lm_pvalue"] < 0.05
    assert res["reject_null_at_05"] is True

def test_durbin_watson_detects_autocorrelation(autocorrelated_model):
    res = run_durbin_watson_test(autocorrelated_model)
    assert res["dw_statistic"] < 1.5
    assert res["serial_correlation_detected"] is True

def test_breusch_godfrey_detects_autocorrelation(autocorrelated_model):
    res = run_breusch_godfrey_test(autocorrelated_model, nlags=2)
    assert res["lm_pvalue"] < 0.05
    assert res["reject_null_at_05"] is True
