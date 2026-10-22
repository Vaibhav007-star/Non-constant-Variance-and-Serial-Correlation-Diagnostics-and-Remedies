/**
 * Econometric Diagnostics & Remedies Web Cockpit
 * Interactive Client Application
 * Author: Vaibhav (@Vaibhav007-star)
 */

// Application State
const state = {
  presets: [],
  currentDatasetId: "macro",
  currentDataset: null,
  targetCol: "",
  featureCols: [],
  addConstant: true,
  alpha: 0.05,
  currentTheme: "obsidian",
  auditData: null,
  activeTab: "tab-baseline"
};

// DOM References
const elements = {
  presetList: document.getElementById("preset-list"),
  dropzone: document.getElementById("csv-dropzone"),
  csvFileInput: document.getElementById("csv-file-input"),
  targetSelect: document.getElementById("target-select"),
  featuresContainer: document.getElementById("features-container"),
  constantToggle: document.getElementById("constant-toggle"),
  alphaSlider: document.getElementById("alpha-slider"),
  alphaValue: document.getElementById("alpha-value"),
  btnRunAudit: document.getElementById("btn-run-audit"),
  themeSelect: document.getElementById("theme-select"),
  tabButtons: document.querySelectorAll(".tab-btn"),
  tabPanes: document.querySelectorAll(".tab-pane"),
  loadingOverlay: document.getElementById("loading-overlay"),
  toastContainer: document.getElementById("toast-container")
};

// ---------------- INITIALIZATION ----------------

document.addEventListener("DOMContentLoaded", async () => {
  setupTheme();
  setupEventListeners();
  await loadPresets();
});

function setupTheme() {
  const savedTheme = localStorage.getItem("econ_theme") || "obsidian";
  state.currentTheme = savedTheme;
  document.documentElement.setAttribute("data-theme", savedTheme);
  elements.themeSelect.value = savedTheme;
}

function switchTheme(newTheme) {
  state.currentTheme = newTheme;
  localStorage.setItem("econ_theme", newTheme);
  document.documentElement.setAttribute("data-theme", newTheme);
  
  // Re-render charts with updated theme colors
  if (state.auditData && state.auditData.charts) {
    renderAllCharts(state.auditData.charts);
  }
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = "toast";
  const icon = type === "success" ? "✅" : type === "error" ? "⚠️" : "ℹ️";
  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

// ---------------- EVENT LISTENERS ----------------

function setupEventListeners() {
  // Theme change
  elements.themeSelect.addEventListener("change", (e) => {
    switchTheme(e.target.value);
  });

  // Alpha slider
  elements.alphaSlider.addEventListener("input", (e) => {
    state.alpha = parseFloat(e.target.value);
    elements.alphaValue.textContent = state.alpha.toFixed(2);
  });

  // Constant toggle
  elements.constantToggle.addEventListener("change", (e) => {
    state.addConstant = e.target.checked;
  });

  // Run audit button
  elements.btnRunAudit.addEventListener("click", () => {
    executeAudit();
  });

  // Keyboard shortcut Ctrl+Enter
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      executeAudit();
    }
  });

  // Tab switching
  elements.tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });

  // Target variable change
  elements.targetSelect.addEventListener("change", (e) => {
    state.targetCol = e.target.value;
    updateFeaturePills();
  });

  // Dropzone drag-and-drop
  elements.dropzone.addEventListener("click", () => elements.csvFileInput.click());
  elements.csvFileInput.addEventListener("change", handleFileUpload);
  
  elements.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    elements.dropzone.classList.add("dragover");
  });
  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.classList.remove("dragover");
  });
  elements.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    elements.dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  // Global window resize — reflow all visible Plotly charts
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const chartIds = ["chart-residuals", "chart-scale-location", "chart-acf", "chart-lag-scatter", "chart-qq", "chart-distribution"];
      chartIds.forEach(id => {
        const el = document.getElementById(id);
        if (el && el.data) Plotly.Plots.resize(el);
      });
    }, 150);
  });
}

function switchTab(tabId) {
  state.activeTab = tabId;
  elements.tabButtons.forEach(b => {
    b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
  });
  elements.tabPanes.forEach(p => {
    p.classList.toggle("active", p.id === tabId);
  });

  // Trigger Plotly relayout on tab switch to avoid chart cutoff
  if (tabId === "tab-visuals") {
    setTimeout(() => {
      const chartIds = ["chart-residuals", "chart-scale-location", "chart-acf", "chart-lag-scatter", "chart-qq", "chart-distribution"];
      chartIds.forEach(id => {
        const el = document.getElementById(id);
        if (el && el.data) {
          Plotly.Plots.resize(el);
        }
      });
    }, 50);
  }
}

// ---------------- PRESET & DATA HANDLING ----------------

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    const data = await res.json();
    state.presets = data.presets || [];
    renderPresetCards();
    
    // Select default preset
    if (state.presets.length > 0) {
      selectDataset(state.presets[0].id);
    }
  } catch (err) {
    showToast("Failed to connect to backend engine: " + err.message, "error");
  }
}

function renderPresetCards() {
  elements.presetList.innerHTML = "";
  state.presets.forEach(preset => {
    const card = document.createElement("div");
    card.className = `preset-card ${preset.id === state.currentDatasetId ? "active" : ""}`;
    card.setAttribute("data-id", preset.id);
    card.innerHTML = `
      <div class="preset-header">
        <span class="preset-title">${preset.name}</span>
        <span class="preset-badge">${preset.badge}</span>
      </div>
      <div class="preset-desc">${preset.description}</div>
    `;
    card.addEventListener("click", () => selectDataset(preset.id));
    elements.presetList.appendChild(card);
  });
}

function selectDataset(datasetId) {
  state.currentDatasetId = datasetId;
  
  // Highlight active card
  document.querySelectorAll(".preset-card").forEach(c => {
    c.classList.toggle("active", c.getAttribute("data-id") === datasetId);
  });
  
  const preset = state.presets.find(p => p.id === datasetId);
  if (preset) {
    state.currentDataset = preset;
    state.targetCol = preset.default_target;
    state.featureCols = [...preset.default_features];
    
    populateTargetDropdown(preset.columns);
    updateFeaturePills();
    executeAudit();
    showToast(`Loaded ${preset.name}`, "info");
  }
}

function populateTargetDropdown(columns) {
  elements.targetSelect.innerHTML = "";
  columns.forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    if (col === state.targetCol) opt.selected = true;
    elements.targetSelect.appendChild(opt);
  });
}

function updateFeaturePills() {
  const columns = (state.currentDataset && state.currentDataset.columns) ? state.currentDataset.columns : [];
  elements.featuresContainer.innerHTML = "";
  
  // Available features exclude target
  const available = columns.filter(c => c !== state.targetCol);
  
  // Clean state featureCols to ensure valid
  state.featureCols = state.featureCols.filter(f => available.includes(f));
  if (state.featureCols.length === 0 && available.length > 0) {
    state.featureCols = [available[0]];
  }

  available.forEach(col => {
    const isSelected = state.featureCols.includes(col);
    const pill = document.createElement("div");
    pill.className = `feature-pill ${isSelected ? "selected" : ""}`;
    pill.innerHTML = `<span>${isSelected ? "✓" : "+"}</span> <span>${col}</span>`;
    pill.addEventListener("click", () => {
      if (state.featureCols.includes(col)) {
        if (state.featureCols.length > 1) {
          state.featureCols = state.featureCols.filter(f => f !== col);
        } else {
          showToast("At least one regressor is required.", "warning");
          return;
        }
      } else {
        state.featureCols.push(col);
      }
      updateFeaturePills();
    });
    elements.featuresContainer.appendChild(pill);
  });
}

async function handleFileUpload(e) {
  if (e.target.files && e.target.files[0]) {
    await uploadFile(e.target.files[0]);
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  
  try {
    showToast(`Uploading ${file.name}...`, "info");
    const res = await fetch("/api/upload-csv", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload error");

    const customPreset = {
      id: data.upload_id,
      name: `Custom: ${data.filename}`,
      badge: `CSV (N=${data.rows})`,
      description: `User-uploaded dataset with ${data.columns.length} numeric columns.`,
      default_target: data.default_target,
      default_features: data.default_features,
      columns: data.columns
    };

    state.presets.push(customPreset);
    renderPresetCards();
    selectDataset(customPreset.id);
    showToast("CSV dataset parsed successfully!", "success");
  } catch (err) {
    showToast("Error processing CSV: " + err.message, "error");
  }
}

// ---------------- RUN AUDIT ----------------

async function executeAudit() {
  if (!state.targetCol || state.featureCols.length === 0) {
    showToast("Please choose a target variable and at least one regressor.", "warning");
    return;
  }

  elements.loadingOverlay.style.display = "flex";
  
  const payload = {
    dataset_id: state.currentDatasetId,
    target_col: state.targetCol,
    feature_cols: state.featureCols,
    add_constant: state.addConstant,
    alpha: state.alpha
  };

  try {
    const res = await fetch("/api/audit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Computation failed");

    state.auditData = data;
    
    // Render all panels
    renderKPICards(data.metrics);
    renderAssumptionBanners(data.diagnostics);
    renderBaselineTable(data.params, data.metrics);
    renderAllCharts(data.charts);
    renderDiagnosticCards(data.diagnostics);
    renderScorecard(data.scorecard);
    renderExecutiveSummary(data.audit_summary);

    showToast("Econometric audit completed!", "success");
  } catch (err) {
    showToast("Audit Error: " + err.message, "error");
  } finally {
    elements.loadingOverlay.style.display = "none";
  }
}

// ---------------- RENDERERS ----------------

function renderKPICards(m) {
  document.getElementById("kpi-nobs").textContent = m.nobs.toLocaleString();
  document.getElementById("kpi-r2").textContent = m.r_squared.toFixed(4);
  document.getElementById("kpi-adj-r2").textContent = m.r_squared_adj.toFixed(4);
  document.getElementById("kpi-se").textContent = m.residual_std_error.toFixed(4);
  document.getElementById("kpi-f").textContent = m.f_statistic.toFixed(2);
  document.getElementById("kpi-dw").textContent = m.durbin_watson.toFixed(3);
}

function renderAssumptionBanners(diag) {
  const hasHet = diag.heteroscedasticity.overall_heteroscedasticity_detected;
  const hasSer = diag.serial_correlation.overall_serial_correlation_detected;
  
  const bp = diag.heteroscedasticity.breusch_pagan;
  const dw = diag.serial_correlation.durbin_watson;
  const bg = diag.serial_correlation.breusch_godfrey;

  // Heteroscedasticity Banner
  const hetBanner = document.getElementById("banner-het");
  if (hasHet) {
    hetBanner.className = "alert-banner alert-danger";
    hetBanner.innerHTML = `
      <div class="banner-icon">🚨</div>
      <div class="banner-content">
        <div class="banner-title">Heteroscedasticity Detected!</div>
        <div class="banner-desc">Non-constant error variance confirmed (Breusch-Pagan p = ${bp.lm_pvalue < 0.0001 ? '<0.0001' : bp.lm_pvalue.toFixed(4)}). OLS standard errors are severely biased; hypothesis tests produce false positives.</div>
        <span class="banner-metric-pill">Recommendation: Use White HC3 or Feasible GLS</span>
      </div>
    `;
  } else {
    hetBanner.className = "alert-banner alert-success";
    hetBanner.innerHTML = `
      <div class="banner-icon">✅</div>
      <div class="banner-content">
        <div class="banner-title">Homoscedasticity Verified</div>
        <div class="banner-desc">Residual variance is constant across fitted values. Gauss-Markov variance assumption holds; classical OLS standard errors are BLUE.</div>
        <span class="banner-metric-pill">OLS Standard Errors are Valid</span>
      </div>
    `;
  }

  // Serial Correlation Banner
  const serBanner = document.getElementById("banner-ser");
  if (hasSer) {
    serBanner.className = "alert-banner alert-warning";
    serBanner.innerHTML = `
      <div class="banner-icon">⚠️</div>
      <div class="banner-content">
        <div class="banner-title">Serial Correlation Detected!</div>
        <div class="banner-desc">Autocorrelation present in regression disturbances (DW = ${dw.dw_statistic.toFixed(3)}, BG p = ${bg.lm_pvalue < 0.0001 ? '<0.0001' : bg.lm_pvalue.toFixed(4)}). Standard errors are underestimated.</div>
        <span class="banner-metric-pill">Recommendation: Apply Newey-West HAC or Prais-Winsten</span>
      </div>
    `;
  } else {
    serBanner.className = "alert-banner alert-success";
    serBanner.innerHTML = `
      <div class="banner-icon">✅</div>
      <div class="banner-content">
        <div class="banner-title">Error Independence Verified</div>
        <div class="banner-desc">No significant residual persistence detected across time lags. Disturbances satisfy independent identically distributed (i.i.d.) conditions.</div>
        <span class="banner-metric-pill">No Dynamic Lag Adjustment Needed</span>
      </div>
    `;
  }
}

function renderBaselineTable(params, metrics) {
  const tbody = document.getElementById("ols-params-tbody");
  tbody.innerHTML = "";

  params.forEach(p => {
    const tr = document.createElement("tr");
    const pValFormatted = p.p_value < 0.0001 ? p.p_value.toExponential(3) : p.p_value.toFixed(4);
    const pBadgeClass = p.p_value < 0.05 ? "p-sig" : "p-insig";

    tr.innerHTML = `
      <td style="font-weight: 600;">${p.variable}</td>
      <td class="mono-val" style="color: var(--accent-cyan);">${p.coeff.toFixed(4)}</td>
      <td class="mono-val">${p.se.toFixed(4)}</td>
      <td class="mono-val">${p.t_stat.toFixed(3)}</td>
      <td><span class="badge-pvalue ${pBadgeClass}">${pValFormatted}</span></td>
      <td class="mono-val" style="font-size: 11px; color: var(--text-dim);">[${p.ci_lower.toFixed(3)}, ${p.ci_upper.toFixed(3)}]</td>
    `;
    tbody.appendChild(tr);
  });

  // Specification stats
  document.getElementById("spec-target").textContent = state.targetCol;
  document.getElementById("spec-df-resid").textContent = metrics.df_resid;
  document.getElementById("spec-aic").textContent = metrics.aic.toFixed(2);
  document.getElementById("spec-bic").textContent = metrics.bic.toFixed(2);
  document.getElementById("spec-llf").textContent = metrics.log_likelihood.toFixed(2);
  document.getElementById("spec-fpval").textContent = metrics.f_pvalue < 0.0001 ? metrics.f_pvalue.toExponential(3) : metrics.f_pvalue.toFixed(4);
}

// ---------------- PLOTLY CHARTS ----------------

function getChartThemeConfig() {
  const isLight = state.currentTheme === "light";
  return {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: {
      family: "Plus Jakarta Sans, sans-serif",
      color: isLight ? "#334155" : "#94A3B8",
      size: 11
    },
    gridcolor: isLight ? "rgba(148, 163, 184, 0.2)" : "rgba(255, 255, 255, 0.07)",
    zerolinecolor: isLight ? "rgba(148, 163, 184, 0.4)" : "rgba(255, 255, 255, 0.15)",
    accentCyan: isLight ? "#0284C7" : "#06B6D4",
    accentRed: "#EF4444",
    accentEmerald: "#10B981",
    accentAmber: "#F59E0B"
  };
}

function renderAllCharts(charts) {
  const cfg = getChartThemeConfig();
  const defaultPlotlyConfig = { responsive: true, displayModeBar: false };
  const ALL_CHART_IDS = ["chart-residuals", "chart-scale-location", "chart-acf", "chart-lag-scatter", "chart-qq", "chart-distribution"];

  // Helper: builds a layout with autosize and shared defaults
  function makeLayout(xTitle, yTitle, extras) {
    return Object.assign({
      autosize: true,
      paper_bgcolor: cfg.paper_bgcolor,
      plot_bgcolor: cfg.plot_bgcolor,
      font: cfg.font,
      margin: { t: 16, r: 16, b: 44, l: 52 },
      xaxis: { title: xTitle, gridcolor: cfg.gridcolor, zerolinecolor: cfg.zerolinecolor, automargin: true },
      yaxis: { title: yTitle, gridcolor: cfg.gridcolor, zerolinecolor: cfg.zerolinecolor, automargin: true },
      showlegend: false
    }, extras || {});
  }

  // 1. Residuals vs Fitted
  const trace1_pts = {
    x: charts.residuals_vs_fitted.fitted,
    y: charts.residuals_vs_fitted.residuals,
    mode: "markers",
    type: "scatter",
    name: "Residuals",
    marker: { color: cfg.accentCyan, size: 6, opacity: 0.65 }
  };
  const trace1_zero = {
    x: [Math.min(...charts.residuals_vs_fitted.fitted), Math.max(...charts.residuals_vs_fitted.fitted)],
    y: [0, 0],
    mode: "lines",
    name: "Zero Line",
    line: { color: cfg.accentRed, dash: "dash", width: 1.5 }
  };
  const trace1_lowess = {
    x: charts.residuals_vs_fitted.lowess_x,
    y: charts.residuals_vs_fitted.lowess_y,
    mode: "lines",
    name: "LOWESS Curve",
    line: { color: "#F43F5E", width: 2.2 }
  };
  const layout1 = makeLayout("Fitted Values (Ŷ)", "Residuals (e)");
  Plotly.newPlot("chart-residuals", [trace1_pts, trace1_zero, trace1_lowess], layout1, defaultPlotlyConfig);

  // 2. Scale-Location
  const trace2_pts = {
    x: charts.scale_location.fitted,
    y: charts.scale_location.sqrt_abs_std,
    mode: "markers",
    type: "scatter",
    name: "√|Std Resid|",
    marker: { color: cfg.accentEmerald, size: 6, opacity: 0.65 }
  };
  const trace2_lowess = {
    x: charts.scale_location.lowess_x,
    y: charts.scale_location.lowess_y,
    mode: "lines",
    name: "Trend",
    line: { color: cfg.accentRed, width: 2.2 }
  };
  const layout2 = makeLayout("Fitted Values (Ŷ)", "√|Standardized Resid|");
  Plotly.newPlot("chart-scale-location", [trace2_pts, trace2_lowess], layout2, defaultPlotlyConfig);

  // 3. Autocorrelation Function (ACF)
  const acf = charts.acf;
  const trace3_bars = {
    x: acf.lags,
    y: acf.acf_values,
    type: "bar",
    name: "Autocorrelation",
    marker: { color: cfg.accentCyan, width: 0.25 }
  };
  const trace3_upper = {
    x: acf.lags,
    y: acf.ci_upper,
    mode: "lines",
    name: "+95% CI",
    line: { color: "#38BDF8", dash: "dot", width: 1 }
  };
  const trace3_lower = {
    x: acf.lags,
    y: acf.ci_lower,
    mode: "lines",
    name: "-95% CI",
    line: { color: "#38BDF8", dash: "dot", width: 1 },
    fill: "tonexty",
    fillcolor: "rgba(56, 189, 248, 0.12)"
  };
  const layout3 = makeLayout("Lag (k)", "ACF (ρₖ)", {
    yaxis: { title: "ACF (ρₖ)", range: [-1.05, 1.05], gridcolor: cfg.gridcolor, zerolinecolor: cfg.zerolinecolor, automargin: true }
  });
  Plotly.newPlot("chart-acf", [trace3_bars, trace3_upper, trace3_lower], layout3, defaultPlotlyConfig);

  // 4. Lag Scatter (e_t vs e_{t-1})
  const lag = charts.lag_scatter;
  const trace4_pts = {
    x: lag.e_tm1,
    y: lag.e_t,
    mode: "markers",
    type: "scatter",
    name: "Residuals",
    marker: { color: cfg.accentAmber, size: 6, opacity: 0.7 }
  };
  const trace4_line = {
    x: lag.fit_x,
    y: lag.fit_y,
    mode: "lines",
    name: `Fit (ρ̂=${lag.slope.toFixed(3)})`,
    line: { color: "#B45309", width: 2.2 }
  };
  const layout4 = makeLayout("Residual at t-1 (eₜ₋₁)", "Residual at t (eₜ)");
  Plotly.newPlot("chart-lag-scatter", [trace4_pts, trace4_line], layout4, defaultPlotlyConfig);

  // 5. Normal Q-Q Plot
  const qq = charts.qq;
  const trace5_pts = {
    x: qq.theoretical,
    y: qq.sample,
    mode: "markers",
    type: "scatter",
    marker: { color: "#A855F7", size: 6, opacity: 0.7 }
  };
  const trace5_line = {
    x: qq.line_x,
    y: qq.line_y,
    mode: "lines",
    line: { color: cfg.accentRed, width: 2 }
  };
  const layout5 = makeLayout("Theoretical Quantiles", "Sample Residual Quantiles");
  Plotly.newPlot("chart-qq", [trace5_pts, trace5_line], layout5, defaultPlotlyConfig);

  // 6. Distribution & Histogram
  const dist = charts.distribution;
  const trace6_hist = {
    x: dist.bin_centers,
    y: dist.density,
    type: "bar",
    marker: { color: "rgba(6, 182, 212, 0.4)", line: { color: cfg.accentCyan, width: 1 } }
  };
  const trace6_norm = {
    x: dist.norm_x,
    y: dist.norm_y,
    mode: "lines",
    line: { color: "#3B82F6", width: 2 }
  };
  const layout6 = makeLayout("Residual Error (e)", "Density");
  Plotly.newPlot("chart-distribution", [trace6_hist, trace6_norm], layout6, defaultPlotlyConfig);

  // Force an immediate resize pass for all charts so Plotly matches the CSS container
  requestAnimationFrame(() => {
    ALL_CHART_IDS.forEach(id => {
      const el = document.getElementById(id);
      if (el && el.data) Plotly.Plots.resize(el);
    });
  });
}

// ---------------- HYPOTHESIS TESTS ----------------

function renderDiagnosticCards(diag) {
  const container = document.getElementById("tests-container");
  container.innerHTML = "";

  const het = diag.heteroscedasticity;
  const ser = diag.serial_correlation;

  const testList = [
    {
      name: "Breusch-Pagan Lagrange Multiplier Test",
      category: "Heteroscedasticity",
      statName: "LM Statistic",
      statVal: het.breusch_pagan.lm_statistic.toFixed(4),
      pVal: het.breusch_pagan.lm_pvalue,
      rejected: het.breusch_pagan.reject_null_at_05,
      verdict: het.breusch_pagan.verdict,
      desc: "Tests whether residual variance depends linearly on the regressors."
    },
    {
      name: "White's General Heteroscedasticity Test",
      category: "Heteroscedasticity",
      statName: "LM Statistic",
      statVal: het.white.lm_statistic.toFixed(4),
      pVal: het.white.lm_pvalue,
      rejected: het.white.reject_null_at_05,
      verdict: het.white.verdict,
      desc: "Tests for arbitrary heteroscedasticity including non-linear squares and interaction cross-products."
    },
    {
      name: "Goldfeld-Quandt Variance Ratio Test",
      category: "Heteroscedasticity",
      statName: "F Statistic",
      statVal: het.goldfeld_quandt.f_statistic.toFixed(4),
      pVal: het.goldfeld_quandt.pvalue,
      rejected: het.goldfeld_quandt.reject_null_at_05,
      verdict: het.goldfeld_quandt.verdict,
      desc: "Splits sample to compare variance ratios between high and low regressor regimes."
    },
    {
      name: "Durbin-Watson Test (First-Order AR(1))",
      category: "Serial Correlation",
      statName: "d Statistic",
      statVal: ser.durbin_watson.dw_statistic.toFixed(4),
      pVal: null,
      rejected: ser.durbin_watson.serial_correlation_detected,
      verdict: ser.durbin_watson.verdict,
      desc: "Benchmarks: d ≈ 2 implies no serial correlation. d < 1.5 indicates positive autocorrelation."
    },
    {
      name: "Breusch-Godfrey LM Test (Higher-Order AR)",
      category: "Serial Correlation",
      statName: "LM Statistic",
      statVal: ser.breusch_godfrey.lm_statistic.toFixed(4),
      pVal: ser.breusch_godfrey.lm_pvalue,
      rejected: ser.breusch_godfrey.reject_null_at_05,
      verdict: ser.breusch_godfrey.verdict,
      desc: "Tests for autoregressive or moving-average error structures up to order p."
    },
    {
      name: "Ljung-Box Portmanteau Autocorrelation Test",
      category: "Serial Correlation",
      statName: "Min p-value",
      statVal: ser.ljung_box.min_pvalue < 0.0001 ? ser.ljung_box.min_pvalue.toExponential(3) : ser.ljung_box.min_pvalue.toFixed(4),
      pVal: ser.ljung_box.min_pvalue,
      rejected: ser.ljung_box.reject_null_at_05,
      verdict: ser.ljung_box.verdict,
      desc: "Examines cumulative autocorrelations across multiple lags simultaneously."
    }
  ];

  testList.forEach(t => {
    const card = document.createElement("div");
    card.className = "test-card";
    const badgeClass = t.rejected ? "badge-rejected" : "badge-accepted";
    const badgeText = t.rejected ? "Violation Detected" : "Null Retained";
    const pvalDisplay = t.pVal !== null ? (t.pVal < 0.0001 ? t.pVal.toExponential(3) : t.pVal.toFixed(4)) : "N/A";

    card.innerHTML = `
      <div class="test-card-top">
        <div>
          <span style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: var(--accent-cyan);">${t.category}</span>
          <div class="test-name">${t.name}</div>
        </div>
        <span class="test-badge ${badgeClass}">${badgeText}</span>
      </div>
      <p style="font-size: 12px; color: var(--text-dim);">${t.desc}</p>
      <div class="test-stats-row">
        <span>${t.statName}: <b class="mono-val">${t.statVal}</b></span>
        <span>p-value: <b class="mono-val">${pvalDisplay}</b></span>
      </div>
      <div class="test-verdict">${t.verdict}</div>
    `;
    container.appendChild(card);
  });
}

// ---------------- SCORECARD & REMEDIES ----------------

function renderScorecard(scorecard) {
  const tbody = document.getElementById("scorecard-tbody");
  tbody.innerHTML = "";
  const metersContainer = document.getElementById("meters-container");
  metersContainer.innerHTML = "";

  scorecard.forEach(row => {
    const tr = document.createElement("tr");
    const fmt = (val) => (val !== null && val !== undefined && !isNaN(val)) ? (val < 0.0001 && val > 0 ? val.toExponential(3) : val.toFixed(4)) : "—";
    
    tr.innerHTML = `
      <td style="font-weight: 700;">${row.Variable}</td>
      <td class="mono-val">${fmt(row.OLS_Coeff)}</td>
      <td class="mono-val" style="color: var(--text-muted);">${fmt(row.OLS_SE)}</td>
      <td class="mono-val">${fmt(row.OLS_pVal)}</td>
      <td class="mono-val" style="color: var(--accent-cyan); font-weight: 600;">${fmt(row.HC3_SE)}</td>
      <td class="mono-val">${fmt(row.HC3_pVal)}</td>
      <td class="mono-val" style="color: var(--accent-blue); font-weight: 600;">${fmt(row.HAC_SE)}</td>
      <td class="mono-val">${fmt(row.HAC_pVal)}</td>
      <td class="mono-val">${fmt(row.FGLS_Coeff)}</td>
      <td class="mono-val" style="color: var(--accent-emerald); font-weight: 600;">${fmt(row.FGLS_SE)}</td>
    `;
    tbody.appendChild(tr);

    // Render Inflation Meter
    if (row.Variable !== "const") {
      const hc3Ratio = row.ratio_hc3_ols || 1.0;
      const hacRatio = row.ratio_hac_ols || 1.0;
      const meterCard = document.createElement("div");
      meterCard.style.padding = "12px 16px";
      meterCard.style.background = "var(--bg-glass)";
      meterCard.style.borderRadius = "10px";
      meterCard.style.border = "1px solid var(--border-glass)";
      meterCard.style.display = "flex";
      meterCard.style.flexDirection = "column";
      meterCard.style.gap = "8px";

      const maxRatio = Math.max(hc3Ratio, hacRatio, 1.0);
      const hc3Pct = Math.min(100, Math.round((hc3Ratio / Math.max(3, maxRatio * 1.2)) * 100));
      const hacPct = Math.min(100, Math.round((hacRatio / Math.max(3, maxRatio * 1.2)) * 100));

      meterCard.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 700;">
          <span>${row.Variable}</span>
          <span class="mono-val" style="color: var(--accent-cyan);">HAC/OLS: ${hacRatio.toFixed(2)}x SE Inflation</span>
        </div>
        <div class="meter-wrapper">
          <span style="font-size: 11px; width: 60px; color: var(--text-dim);">HC3</span>
          <div class="meter-track">
            <div class="meter-fill" style="width: ${hc3Pct}%; background: var(--accent-cyan);"></div>
          </div>
          <span class="mono-val" style="font-size: 11px; width: 45px; text-align: right;">${hc3Ratio.toFixed(2)}x</span>
        </div>
        <div class="meter-wrapper">
          <span style="font-size: 11px; width: 60px; color: var(--text-dim);">HAC</span>
          <div class="meter-track">
            <div class="meter-fill" style="width: ${hacPct}%; background: var(--accent-blue);"></div>
          </div>
          <span class="mono-val" style="font-size: 11px; width: 45px; text-align: right;">${hacRatio.toFixed(2)}x</span>
        </div>
      `;
      metersContainer.appendChild(meterCard);
    }
  });
}

// ---------------- EXECUTIVE SUMMARY & EXPORT ----------------

function renderExecutiveSummary(summary) {
  document.getElementById("summary-verdict").textContent = summary.verdict;
  const list = document.getElementById("recommendations-list");
  list.innerHTML = "";
  summary.recommendations.forEach(r => {
    const li = document.createElement("li");
    li.style.marginBottom = "8px";
    li.style.color = "var(--text-muted)";
    li.innerHTML = `📌 <b style="color: var(--text-main);">${r}</b>`;
    list.appendChild(li);
  });
}

// Export Handlers
window.downloadReportMarkdown = async function() {
  if (!state.auditData) return;
  const s = state.auditData;
  const m = s.metrics;
  const md = `# Econometric Audit & Diagnostics Report
Generated: ${new Date().toLocaleString()}
Target Variable: ${s.model_specs.target_col}
Regressors: ${s.model_specs.feature_cols.join(", ")}
Observations: ${m.nobs}

---

## 1. Baseline Performance
- R-squared: ${m.r_squared.toFixed(4)} (Adj R²: ${m.r_squared_adj.toFixed(4)})
- F-Statistic: ${m.f_statistic.toFixed(2)} (p = ${m.f_pvalue.toExponential(3)})
- Residual SE: ${m.residual_std_error.toFixed(4)}
- Durbin-Watson: ${m.durbin_watson.toFixed(4)}

## 2. Diagnostics
- Heteroscedasticity: ${s.diagnostics.heteroscedasticity.breusch_pagan.verdict}
- Serial Correlation: ${s.diagnostics.serial_correlation.durbin_watson.verdict}

## 3. Executive Policy Verdict
${s.audit_summary.verdict}
${s.audit_summary.recommendations.map(r => "- " + r).join("\n")}
`;

  try {
    const res = await fetch("/api/export-markdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ markdown_text: md, filename: `econometric_audit_${state.targetCol}.md` })
    });
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `econometric_audit_${state.targetCol}.md`;
    a.click();
    showToast("Audit report downloaded!", "success");
  } catch (err) {
    showToast("Download failed: " + err.message, "error");
  }
};

window.downloadScorecardCsv = async function() {
  if (!state.auditData || !state.auditData.scorecard) return;
  try {
    const res = await fetch("/api/export-csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: state.auditData.scorecard, filename: `scorecard_${state.targetCol}.csv` })
    });
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scorecard_${state.targetCol}.csv`;
    a.click();
    showToast("Scorecard CSV downloaded!", "success");
  } catch (err) {
    showToast("Download failed: " + err.message, "error");
  }
};

