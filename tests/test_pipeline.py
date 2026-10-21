"""
Integration tests for EconometricAuditEngine pipeline and scorecard.
"""

import pytest
import pandas as pd
import numpy as np
from src.pipeline.econometric_engine import EconometricAuditEngine

@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 150
    x1 = np.random.uniform(10, 50, n)
    x2 = np.random.uniform(1, 10, n)
    e = np.random.normal(0, 0.5 * x2)
    y = 5.0 + 2.0 * x1 - 1.5 * x2 + e
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

def test_audit_engine_end_to_end(sample_dataset):
    engine = EconometricAuditEngine(
        data=sample_dataset,
        target_col="y",
        feature_cols=["x1", "x2"],
        add_constant=True
    )
    
    # 1. Baseline
    model = engine.fit_baseline()
    assert model is not None
    assert int(model.nobs) == 150
    assert len(model.params) == 3
    
    # 2. Diagnostics
    diag = engine.run_diagnostics()
    assert "heteroscedasticity" in diag
    assert "serial_correlation" in diag
    assert "overall_heteroscedasticity_detected" in diag["heteroscedasticity"]
    assert "overall_serial_correlation_detected" in diag["serial_correlation"]
    
    # 3. Remedies
    remedies = engine.run_remedies()
    assert "OLS_HC3" in remedies
    assert "OLS_HAC_NeweyWest" in remedies
    
    # 4. Scorecard
    scorecard = engine.generate_scorecard()
    assert isinstance(scorecard, pd.DataFrame)
    assert len(scorecard) == 3
    assert "OLS_Coeff" in scorecard.columns
    assert "HC3_SE" in scorecard.columns
    assert "HAC_SE" in scorecard.columns
    
    # 5. Audit Summary
    summary = engine.generate_audit_summary()
    assert "verdict" in summary
    assert isinstance(summary["recommendations"], list)
    assert len(summary["recommendations"]) > 0
    assert summary["nobs"] == 150
