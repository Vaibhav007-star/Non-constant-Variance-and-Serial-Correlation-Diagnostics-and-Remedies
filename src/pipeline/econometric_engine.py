"""
Unified Econometric Audit & Remedy Engine.
Orchestrates:
1. Baseline OLS Estimation
2. Heteroscedasticity Diagnostics (Breusch-Pagan, White, Goldfeld-Quandt)
3. Serial Correlation Diagnostics (Durbin-Watson, Breusch-Godfrey, Ljung-Box)
4. Comprehensive Remedy Suite:
   - Robust Covariance (HC0, HC1, HC3, HAC Newey-West)
   - Feasible Generalized Least Squares (FGLS)
   - Prais-Winsten / Cochrane-Orcutt for AR(1)
5. Comprehensive Model Scorecard & Executive Decision Report
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.diagnostics.heteroscedasticity import run_all_heteroscedasticity_diagnostics
from src.diagnostics.serial_correlation import run_all_serial_correlation_diagnostics
from src.remedies.robust_standard_errors import fit_ols_with_robust_se
from src.remedies.gls_wls import fit_feasible_gls
from src.remedies.time_series_remedies import prais_winsten, cochrane_orcutt

class EconometricAuditEngine:
    def __init__(self, data: pd.DataFrame, target_col: str, feature_cols: list, add_constant: bool = True):
        self.data = data.dropna(subset=[target_col] + feature_cols).copy()
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.add_constant = add_constant
        
        self.y = self.data[target_col].astype(float)
        X_raw = self.data[feature_cols].astype(float)
        self.X = sm.add_constant(X_raw) if add_constant else X_raw
        
        # Placeholders
        self.baseline_ols = None
        self.hetero_results = None
        self.serial_results = None
        self.scorecard = None
        self.remedies = {}

    def fit_baseline(self):
        """Fits baseline OLS with classical standard errors."""
        self.baseline_ols = sm.OLS(self.y, self.X).fit()
        return self.baseline_ols

    def run_diagnostics(self):
        """Runs the full diagnostic testing suite."""
        if self.baseline_ols is None:
            self.fit_baseline()
            
        self.hetero_results = run_all_heteroscedasticity_diagnostics(self.baseline_ols)
        self.serial_results = run_all_serial_correlation_diagnostics(self.baseline_ols)
        return {
            "heteroscedasticity": self.hetero_results,
            "serial_correlation": self.serial_results
        }

    def run_remedies(self):
        """Estimates corrected models across all major econometric remedy paradigms."""
        if self.baseline_ols is None:
            self.fit_baseline()
            
        # 1. White HC3 Robust Covariance
        hc3_model = fit_ols_with_robust_se(self.y, self.X, cov_type="HC3")
        self.remedies["OLS_HC3"] = hc3_model
        
        # 2. Newey-West HAC Robust Covariance (Robust to both)
        hac_model = fit_ols_with_robust_se(self.y, self.X, cov_type="HAC")
        self.remedies["OLS_HAC_NeweyWest"] = hac_model
        
        # 3. Feasible GLS (Heteroscedasticity remedy)
        try:
            fgls_model, weights, _ = fit_feasible_gls(self.y, self.X)
            self.remedies["Feasible_GLS"] = fgls_model
        except Exception:
            self.remedies["Feasible_GLS"] = None
            
        # 4. Prais-Winsten (Autocorrelation remedy)
        try:
            pw_res = prais_winsten(self.y, self.X)
            self.remedies["Prais_Winsten"] = pw_res
        except Exception:
            self.remedies["Prais_Winsten"] = None

        return self.remedies

    def generate_scorecard(self):
        """Compiles a unified comparative scorecard of coefficients, SEs, and p-values."""
        if not self.remedies:
            self.run_remedies()
            
        cols = list(self.X.columns)
        rows = []
        for idx, var in enumerate(cols):
            # Baseline OLS
            ols_b = float(self.baseline_ols.params.iloc[idx] if hasattr(self.baseline_ols.params, 'iloc') else self.baseline_ols.params[idx])
            ols_se = float(self.baseline_ols.bse.iloc[idx] if hasattr(self.baseline_ols.bse, 'iloc') else self.baseline_ols.bse[idx])
            ols_pval = float(self.baseline_ols.pvalues.iloc[idx] if hasattr(self.baseline_ols.pvalues, 'iloc') else self.baseline_ols.pvalues[idx])
            
            # HC3
            hc3_bse = self.remedies["OLS_HC3"].bse
            hc3_pvals = self.remedies["OLS_HC3"].pvalues
            hc3_se = float(hc3_bse.iloc[idx] if hasattr(hc3_bse, 'iloc') else hc3_bse[idx])
            hc3_pval = float(hc3_pvals.iloc[idx] if hasattr(hc3_pvals, 'iloc') else hc3_pvals[idx])
            
            # HAC Newey-West
            hac_bse = self.remedies["OLS_HAC_NeweyWest"].bse
            hac_pvals = self.remedies["OLS_HAC_NeweyWest"].pvalues
            hac_se = float(hac_bse.iloc[idx] if hasattr(hac_bse, 'iloc') else hac_bse[idx])
            hac_pval = float(hac_pvals.iloc[idx] if hasattr(hac_pvals, 'iloc') else hac_pvals[idx])
            
            # FGLS
            fgls_m = self.remedies.get("Feasible_GLS")
            if fgls_m is not None:
                fgls_b = float(fgls_m.params.iloc[idx] if hasattr(fgls_m.params, 'iloc') else fgls_m.params[idx])
                fgls_se = float(fgls_m.bse.iloc[idx] if hasattr(fgls_m.bse, 'iloc') else fgls_m.bse[idx])
                fgls_pval = float(fgls_m.pvalues.iloc[idx] if hasattr(fgls_m.pvalues, 'iloc') else fgls_m.pvalues[idx])
            else:
                fgls_b, fgls_se, fgls_pval = np.nan, np.nan, np.nan

            rows.append({
                "Variable": var,
                "OLS_Coeff": ols_b,
                "OLS_SE": ols_se,
                "OLS_pVal": ols_pval,
                "HC3_SE": hc3_se,
                "HC3_pVal": hc3_pval,
                "HAC_SE": hac_se,
                "HAC_pVal": hac_pval,
                "FGLS_Coeff": fgls_b,
                "FGLS_SE": fgls_se,
                "FGLS_pVal": fgls_pval
            })
            
        self.scorecard = pd.DataFrame(rows)
        return self.scorecard

    def generate_audit_summary(self):
        """Produces a structured plain-English diagnosis and executive recommendation."""
        if self.hetero_results is None or self.serial_results is None:
            self.run_diagnostics()
            
        has_hetero = self.hetero_results["overall_heteroscedasticity_detected"]
        has_serial = self.serial_results["overall_serial_correlation_detected"]
        
        recommendations = []
        if has_hetero and has_serial:
            verdict = "CRITICAL: Both Heteroscedasticity AND Serial Correlation are present."
            recommendations.append("Standard OLS standard errors are biased and severely unreliable.")
            recommendations.append("PRIMARY REMEDY: Use Newey-West HAC Standard Errors (adjusts simultaneously for both violations).")
            recommendations.append("ALTERNATIVE: Estimate Prais-Winsten / Dynamic Lag model with robust covariance.")
        elif has_hetero and not has_serial:
            verdict = "Heteroscedasticity Detected (Constant Variance Assumption Violated)."
            recommendations.append("OLS estimates remain unbiased but are no longer BLUE (Best Linear Unbiased Estimator).")
            recommendations.append("PRIMARY REMEDY: Use MacKinnon-White HC3 Robust Standard Errors.")
            recommendations.append("EFFICIENCY REMEDY: Implement Weighted Least Squares (WLS) or Feasible GLS if variance structure can be modeled.")
        elif not has_hetero and has_serial:
            verdict = "Serial Correlation (Autocorrelation) Detected."
            recommendations.append("Residuals display temporal persistence violating error independence.")
            recommendations.append("PRIMARY REMEDY: Apply Prais-Winsten or Cochrane-Orcutt quasi-differencing.")
            recommendations.append("COVARIANCE REMEDY: Apply Newey-West HAC standard errors.")
        else:
            verdict = "Clean: No Significant Violations of Homoscedasticity or Error Independence Detected."
            recommendations.append("Baseline OLS satisfies Gauss-Markov conditions. Classical standard errors and t-tests are fully valid.")
            
        return {
            "verdict": verdict,
            "has_heteroscedasticity": has_hetero,
            "has_serial_correlation": has_serial,
            "recommendations": recommendations,
            "r_squared": float(self.baseline_ols.rsquared),
            "adj_r_squared": float(self.baseline_ols.rsquared_adj),
            "nobs": int(self.baseline_ols.nobs),
            "aic": float(self.baseline_ols.aic),
            "bic": float(self.baseline_ols.bic)
        }
