"""
Production FastAPI Web Server for Econometric Diagnostics & Remedies Cockpit.
Author: Vaibhav (@Vaibhav007-star)
"""

import os
import sys
import uuid
import io
import json
from typing import List, Optional, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf
from scipy import stats

from src.pipeline.econometric_engine import EconometricAuditEngine

app = FastAPI(
    title="Econometric Diagnostics & Remedies Web Platform",
    description="Full-scale interactive econometric audit suite for Gauss-Markov assumption violations",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# In-memory storage for uploaded datasets
UPLOAD_CACHE: Dict[str, pd.DataFrame] = {}

PRESET_CONFIGS = {
    "macro": {
        "id": "macro",
        "name": "Macroeconomic Time-Series (Serial Correlation & ARCH)",
        "file": "macroeconomic_ts.csv",
        "badge": "Time Series (N=200)",
        "default_target": "GDP_Growth",
        "default_features": ["Interest_Rate", "Money_Growth", "Unemployment_Rate"],
        "description": "Simulated macroeconomic quarterly series exhibiting persistent AR(1) disturbances and ARCH volatility clusters."
    },
    "housing": {
        "id": "housing",
        "name": "Housing Prices Cross-Section (Heteroscedasticity)",
        "file": "housing_prices.csv",
        "badge": "Cross-Section (N=500)",
        "default_target": "Price",
        "default_features": ["Square_Feet", "Lot_Size", "Bedrooms", "Age_Years", "School_Rating"],
        "description": "Cross-sectional residential property sales with variance expanding rapidly as home size and value escalate (fan-shaped heteroscedasticity)."
    },
    "wages": {
        "id": "wages",
        "name": "Corporate Wage & Human Capital (Variance Divergence)",
        "file": "corporate_wages.csv",
        "badge": "Labor Economics (N=300)",
        "default_target": "Hourly_Wage",
        "default_features": ["Education_Years", "Experience_Years", "Tenure_Months", "Firm_Size"],
        "description": "Mincer wage equation displaying escalating earnings dispersion at elevated education and seniority levels."
    }
}

def load_preset_dataframe(preset_id: str) -> pd.DataFrame:
    if preset_id not in PRESET_CONFIGS:
        raise HTTPException(status_code=404, detail="Preset dataset not found.")
    filename = PRESET_CONFIGS[preset_id]["file"]
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File {filename} not found on disk.")
    return pd.read_csv(filepath)

def clean_for_json(obj: Any) -> Any:
    """Sanitizes floating numbers, numpy types, and NaN for strict JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), 6)
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return round(float(val), 6) if isinstance(val, float) else int(val)
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    if isinstance(obj, pd.Series):
        return clean_for_json(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return clean_for_json(obj.to_dict(orient="records"))
    return obj


# ---------------- API ROUTES ----------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Econometric Diagnostics Web API", "version": "2.0.0"}

@app.get("/api/presets")
def get_presets():
    presets_data = []
    for pid, cfg in PRESET_CONFIGS.items():
        try:
            df = load_preset_dataframe(pid)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            preview = df.head(5).to_dict(orient="records")
            presets_data.append({
                "id": pid,
                "name": cfg["name"],
                "badge": cfg["badge"],
                "description": cfg["description"],
                "default_target": cfg["default_target"],
                "default_features": cfg["default_features"],
                "rows": int(len(df)),
                "columns": num_cols,
                "all_columns": df.columns.tolist(),
                "preview": preview
            })
        except Exception as e:
            continue
    return {"presets": presets_data}

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8", errors="replace")))
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise HTTPException(status_code=400, detail="CSV does not contain any numeric columns.")
        
        upload_id = str(uuid.uuid4())[:8]
        UPLOAD_CACHE[upload_id] = df
        
        default_target = num_cols[0]
        default_features = num_cols[1:5] if len(num_cols) > 1 else [num_cols[0]]
        
        return {
            "upload_id": upload_id,
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": num_cols,
            "default_target": default_target,
            "default_features": default_features,
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


class AuditRequest(BaseModel):
    dataset_id: str
    target_col: str
    feature_cols: List[str]
    add_constant: bool = True
    alpha: float = 0.05

@app.post("/api/audit")
def run_audit(req: AuditRequest):
    # Resolve dataframe
    if req.dataset_id in PRESET_CONFIGS:
        df = load_preset_dataframe(req.dataset_id)
    elif req.dataset_id in UPLOAD_CACHE:
        df = UPLOAD_CACHE[req.dataset_id]
    else:
        raise HTTPException(status_code=404, detail="Dataset not found or session expired.")
    
    # Validation
    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_col}' not found in dataset.")
    for f in req.feature_cols:
        if f not in df.columns:
            raise HTTPException(status_code=400, detail=f"Feature regressor '{f}' not found in dataset.")
    if not req.feature_cols:
        raise HTTPException(status_code=400, detail="At least one independent regressor must be specified.")

    try:
        engine = EconometricAuditEngine(
            data=df,
            target_col=req.target_col,
            feature_cols=req.feature_cols,
            add_constant=req.add_constant
        )
        
        ols_model = engine.fit_baseline()
        diag_results = engine.run_diagnostics()
        engine.run_remedies()
        scorecard_df = engine.generate_scorecard()
        audit_summary = engine.generate_audit_summary()
        
        # 1. Model Baseline Summary
        conf_int = ols_model.conf_int()
        param_table = []
        for idx, var in enumerate(ols_model.params.index):
            param_table.append({
                "variable": str(var),
                "coeff": float(ols_model.params.iloc[idx]),
                "se": float(ols_model.bse.iloc[idx]),
                "t_stat": float(ols_model.tvalues.iloc[idx]),
                "p_value": float(ols_model.pvalues.iloc[idx]),
                "ci_lower": float(conf_int.iloc[idx, 0]),
                "ci_upper": float(conf_int.iloc[idx, 1]),
                "significant": bool(ols_model.pvalues.iloc[idx] < req.alpha)
            })

        # 2. Performance Metrics
        dw_val = float(diag_results["serial_correlation"]["durbin_watson"]["dw_statistic"])
        metrics = {
            "nobs": int(ols_model.nobs),
            "df_resid": int(ols_model.df_resid),
            "df_model": int(ols_model.df_model),
            "r_squared": float(ols_model.rsquared),
            "r_squared_adj": float(ols_model.rsquared_adj),
            "f_statistic": float(ols_model.fvalue),
            "f_pvalue": float(ols_model.f_pvalue),
            "residual_std_error": float(np.sqrt(ols_model.scale)),
            "log_likelihood": float(ols_model.llf),
            "aic": float(ols_model.aic),
            "bic": float(ols_model.bic),
            "durbin_watson": dw_val
        }

        # 3. Chart Data Extraction
        fitted = np.asarray(ols_model.fittedvalues)
        resid = np.asarray(ols_model.resid)
        std_resid = resid / (np.std(resid, ddof=1) or 1e-8)
        sqrt_abs_std = np.sqrt(np.abs(std_resid))
        
        # Sort for smooth LOWESS plotting
        sort_idx = np.argsort(fitted)
        fitted_sorted = fitted[sort_idx]
        resid_sorted = resid[sort_idx]
        sqrt_abs_std_sorted = sqrt_abs_std[sort_idx]
        
        # LOWESS for Residuals vs Fitted
        try:
            low_res = lowess(resid_sorted, fitted_sorted, frac=0.6)
            lowess_res_x = low_res[:, 0].tolist()
            lowess_res_y = low_res[:, 1].tolist()
        except Exception:
            lowess_res_x, lowess_res_y = [], []
            
        # LOWESS for Scale-Location
        try:
            low_sl = lowess(sqrt_abs_std_sorted, fitted_sorted, frac=0.6)
            lowess_sl_x = low_sl[:, 0].tolist()
            lowess_sl_y = low_sl[:, 1].tolist()
        except Exception:
            lowess_sl_x, lowess_sl_y = [], []

        # ACF
        nlags = min(20, max(5, len(resid) // 3))
        try:
            acf_vals, confint = acf(resid, nlags=nlags, alpha=req.alpha, result_object=False)
            acf_lags = list(range(len(acf_vals)))
            acf_points = acf_vals.tolist()
            acf_ci_lower = (confint[:, 0] - acf_vals).tolist()
            acf_ci_upper = (confint[:, 1] - acf_vals).tolist()
        except Exception:
            acf_lags, acf_points, acf_ci_lower, acf_ci_upper = [], [], [], []

        # Lag Scatter (e_t vs e_{t-1})
        e_t = resid[1:]
        e_tm1 = resid[:-1]
        if len(e_tm1) > 1:
            m_slope, b_intercept = np.polyfit(e_tm1, e_t, 1)
            min_e = float(np.min(e_tm1))
            max_e = float(np.max(e_tm1))
            lag_fit_x = [min_e, max_e]
            lag_fit_y = [float(m_slope * min_e + b_intercept), float(m_slope * max_e + b_intercept)]
        else:
            m_slope, b_intercept, lag_fit_x, lag_fit_y = 0.0, 0.0, [], []

        # Normal Q-Q Plot
        try:
            (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
            qq_theoretical = osm.tolist()
            qq_sample = osr.tolist()
            min_q = float(np.min(osm))
            max_q = float(np.max(osm))
            qq_line_x = [min_q, max_q]
            qq_line_y = [float(slope * min_q + intercept), float(slope * max_q + intercept)]
        except Exception:
            qq_theoretical, qq_sample, qq_line_x, qq_line_y = [], [], [], []

        # Histogram / Density
        hist_counts, bin_edges = np.histogram(resid, bins=min(30, max(10, len(resid)//10)), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        norm_curve_x = np.linspace(np.min(resid), np.max(resid), 80).tolist()
        norm_curve_y = stats.norm.pdf(norm_curve_x, loc=np.mean(resid), scale=np.std(resid)).tolist()

        chart_data = {
            "residuals_vs_fitted": {
                "fitted": fitted.tolist(),
                "residuals": resid.tolist(),
                "lowess_x": lowess_res_x,
                "lowess_y": lowess_res_y
            },
            "scale_location": {
                "fitted": fitted.tolist(),
                "sqrt_abs_std": sqrt_abs_std.tolist(),
                "lowess_x": lowess_sl_x,
                "lowess_y": lowess_sl_y
            },
            "acf": {
                "lags": acf_lags,
                "acf_values": acf_points,
                "ci_lower": acf_ci_lower,
                "ci_upper": acf_ci_upper
            },
            "lag_scatter": {
                "e_tm1": e_tm1.tolist(),
                "e_t": e_t.tolist(),
                "slope": float(m_slope),
                "intercept": float(b_intercept),
                "fit_x": lag_fit_x,
                "fit_y": lag_fit_y
            },
            "qq": {
                "theoretical": qq_theoretical,
                "sample": qq_sample,
                "line_x": qq_line_x,
                "line_y": qq_line_y
            },
            "distribution": {
                "bin_centers": bin_centers.tolist(),
                "density": hist_counts.tolist(),
                "norm_x": norm_curve_x,
                "norm_y": norm_curve_y
            }
        }

        # 4. Scorecard with Inflation Ratios
        scorecard_records = scorecard_df.to_dict(orient="records")
        for row in scorecard_records:
            ols_se = row.get("OLS_SE") or 1e-8
            row["ratio_hc3_ols"] = (row.get("HC3_SE") or 0.0) / ols_se
            row["ratio_hac_ols"] = (row.get("HAC_SE") or 0.0) / ols_se

        response_payload = {
            "status": "success",
            "model_specs": {
                "target_col": req.target_col,
                "feature_cols": req.feature_cols,
                "add_constant": req.add_constant,
                "alpha": req.alpha
            },
            "metrics": metrics,
            "params": param_table,
            "diagnostics": diag_results,
            "scorecard": scorecard_records,
            "charts": chart_data,
            "audit_summary": audit_summary
        }

        return JSONResponse(content=clean_for_json(response_payload))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Econometric computation error: {str(e)}")


class ExportRequest(BaseModel):
    markdown_text: str
    filename: Optional[str] = "econometric_audit_report.md"

@app.post("/api/export-markdown")
def export_markdown(req: ExportRequest):
    return Response(
        content=req.markdown_text,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{req.filename}"'}
    )

class ExportCsvRequest(BaseModel):
    rows: List[Dict[str, Any]]
    filename: Optional[str] = "econometric_scorecard.csv"

@app.post("/api/export-csv")
def export_csv(req: ExportCsvRequest):
    df_out = pd.DataFrame(req.rows)
    stream = io.StringIO()
    df_out.to_csv(stream, index=False)
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{req.filename}"'}
    )


# Serve Static Assets and SPA
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h3>Web Cockpit is compiling... Please refresh in a moment.</h3>")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)

