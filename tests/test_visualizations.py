"""
Unit tests for econometric visual diagnostic plots.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src.visualizations.diagnostic_plots import (
    plot_residuals_vs_fitted,
    plot_scale_location,
    plot_qq,
    plot_autocorrelation,
    plot_lag_scatter
)

@pytest.fixture
def fitted_model():
    np.random.seed(999)
    x = np.linspace(1, 10, 100)
    y = 2.0 + 1.5 * x + np.random.normal(0, 1, 100)
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()

def test_plot_residuals_vs_fitted(fitted_model):
    fig = plot_residuals_vs_fitted(fitted_model)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)

def test_plot_scale_location(fitted_model):
    fig = plot_scale_location(fitted_model)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)

def test_plot_qq(fitted_model):
    fig = plot_qq(fitted_model)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)

def test_plot_autocorrelation(fitted_model):
    fig = plot_autocorrelation(fitted_model, nlags=10)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)

def test_plot_lag_scatter(fitted_model):
    fig = plot_lag_scatter(fitted_model)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)
