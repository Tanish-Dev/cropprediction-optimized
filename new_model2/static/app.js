/* ============================================================================
   new_model2 dashboard — frontend logic  (multi-model, light theme)
   ========================================================================== */
let META = null;
let HISTORY = null;
let manualDirty = false;
let SELECTED = null;             // active model key
const CHARTS = {};               // id -> Chart instance (for re-render)

Chart.defaults.color = "#64748b";
Chart.defaults.font.family = "Inter, sans-serif";
Chart.defaults.borderColor = "rgba(15,23,42,0.08)";

const fmt = (n, d = 0) =>
  n === null || n === undefined || Number.isNaN(n)
    ? "—"
    : Number(n).toLocaleString(undefined, { maximumFractionDigits: d });

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

function makeChart(id, config) {
  if (CHARTS[id]) CHARTS[id].destroy();
  CHARTS[id] = new Chart(document.getElementById(id), config);
}

function modelInfo(key) {
  return META.models.find((m) => m.key === key);
}

/* ========================================================================== */
async function boot() {
  try {
    META = await getJSON("/api/meta");
    SELECTED = META.default_model;
    document.getElementById("kpiModels").textContent = META.n_models;
    document.getElementById("specTest").textContent = fmt(META.n_test) + " samples (2011–2013)";
    populateSelects(META);
    buildModelPicker(META);
    buildAdvFields(META);
    await refreshModelView();      // loads findings + renders everything for SELECTED
  } catch (e) {
    console.error(e);
  } finally {
    if (window.lucide) lucide.createIcons();
  }
}

function populateSelects(m) {
  const c = document.getElementById("selCountry");
  const k = document.getElementById("selCrop");
  m.countries.forEach((x) => c.add(new Option(x, x)));
  m.crops.forEach((x) => k.add(new Option(x, x)));
  if (m.countries.includes("India")) c.value = "India";
  if (m.crops.includes("Maize")) k.value = "Maize";
}

function buildModelPicker(m) {
  const sel = document.getElementById("selModel");
  m.models.forEach((mo) =>
    sel.add(new Option(`${mo.name}  —  R² ${mo.r2.toFixed(3)}`, mo.key)));
  sel.value = SELECTED;
  sel.addEventListener("change", async () => {
    SELECTED = sel.value;
    await refreshModelView();
    if (window.lucide) lucide.createIcons();
  });
}

function buildAdvFields(m) {
  document.getElementById("advFields").innerHTML = Object.entries(m.editable)
    .map(([col, label]) => {
      const mean = m.feature_stats[col].mean;
      return `<div class="field"><label>${label}</label>
        <input data-col="${col}" type="number" step="any" value="${mean.toFixed(2)}" /></div>`;
    })
    .join("");
}

/* ---- update everything for the selected model ----------------------------- */
async function refreshModelView() {
  const info = modelInfo(SELECTED);
  // header bar + KPIs + tester label
  document.getElementById("mbName").textContent = info.name;
  document.getElementById("mbR2").textContent = info.r2.toFixed(3);
  document.getElementById("mbAcc").textContent = info.acc20.toFixed(1) + "%";
  document.getElementById("kpiR2").textContent = info.r2.toFixed(3);
  document.getElementById("kpiAcc").textContent = info.acc20.toFixed(1) + "%";
  document.getElementById("kpiRmse").textContent = fmt(info.rmse);
  document.getElementById("bestNameHint").textContent = info.name;
  document.getElementById("testerModelName").textContent = info.name;

  renderLeaderboard(META.models, SELECTED);

  const f = await getJSON(`/api/findings?model=${encodeURIComponent(SELECTED)}`);
  renderCharts(f);
  renderAccuracy(f.accuracy);
  renderCountryTable(f.per_country);
}

/* ========================================================================== */
/*  Charts                                                                     */
/* ========================================================================== */
function renderLeaderboard(models, selectedKey) {
  makeChart("leaderboardChart", {
    type: "bar",
    data: {
      labels: models.map((d) => d.name.replace(/^Option [A-C]: /, "")),
      datasets: [{
        label: "Test R²",
        data: models.map((d) => d.r2),
        backgroundColor: models.map((d) =>
          d.key === selectedKey ? "rgba(13,148,136,0.95)" : "rgba(37,99,235,0.45)"),
        borderColor: models.map((d) =>
          d.key === selectedKey ? "rgba(13,148,136,1)" : "transparent"),
        borderWidth: models.map((d) => (d.key === selectedKey ? 2 : 0)),
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y", maintainAspectRatio: false,
      scales: { x: { min: 0.95, max: 1.0, title: { display: true, text: "Test R²" } } },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: {
          label: (c) => {
            const d = models[c.dataIndex];
            return ` R² ${d.r2.toFixed(4)} · RMSE ${fmt(d.rmse)} · ±20% ${d.acc20.toFixed(1)}%`;
          },
        } },
      },
    },
  });
}

function renderCharts(f) {
  makeChart("scatterChart", {
    type: "scatter",
    data: {
      datasets: [
        { label: "Test samples", data: f.scatter, pointRadius: 2.5,
          pointHoverRadius: 5, backgroundColor: "rgba(13,148,136,0.45)" },
        { label: "Perfect prediction", type: "line",
          data: [{ x: 0, y: 0 }, { x: f.diag_max, y: f.diag_max }],
          borderColor: "rgba(124,58,237,0.85)", borderDash: [7, 6],
          borderWidth: 2, pointRadius: 0 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Actual yield (hg/ha)" } },
        y: { title: { display: true, text: "Predicted yield (hg/ha)" } },
      },
      plugins: { legend: { labels: { usePointStyle: true } } },
    },
  });

  makeChart("trendChart", {
    type: "line",
    data: {
      labels: f.trend.years,
      datasets: [
        line("Actual", f.trend.actual, "#0d9488"),
        line("Predicted", f.trend.pred, "#7c3aed", true),
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: { y: { title: { display: true, text: "Mean yield" } } },
      plugins: { legend: { labels: { usePointStyle: true } } },
    },
  });

  const pc = f.per_crop;
  makeChart("cropChart", {
    type: "bar",
    data: {
      labels: pc.map((d) => d.crop),
      datasets: [{
        label: "Test R²", data: pc.map((d) => d.r2),
        backgroundColor: pc.map((d) =>
          d.r2 >= 0.8 ? "rgba(22,163,74,0.78)"
          : d.r2 >= 0.5 ? "rgba(217,119,6,0.78)"
          : "rgba(220,38,38,0.78)"),
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y", maintainAspectRatio: false,
      scales: { x: { min: 0, max: 1 } },
      plugins: { legend: { display: false } },
    },
  });

  const r = f.residuals;
  makeChart("resChart", {
    type: "bar",
    data: {
      labels: r.centers.map((c) => fmt(c)),
      datasets: [{ label: "Count", data: r.counts,
        backgroundColor: "rgba(37,99,235,0.6)", borderRadius: 3 }],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Residual (pred − actual)" }, ticks: { maxTicksLimit: 8 } },
        y: { title: { display: true, text: "Count" } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function line(label, data, color, dashed = false) {
  return {
    label, data, borderColor: color, backgroundColor: color + "22",
    borderWidth: 2.5, borderDash: dashed ? [6, 5] : [], tension: 0.35,
    pointRadius: 2, fill: false,
  };
}

function renderAccuracy(acc) {
  document.getElementById("gaugeVal").textContent = acc.within_20.toFixed(0) + "%";
  const v = acc.within_20;
  makeChart("accGauge", {
    type: "doughnut",
    data: { datasets: [{
      data: [v, 100 - v],
      backgroundColor: ["rgba(13,148,136,0.9)", "rgba(15,23,42,0.07)"],
      borderWidth: 0, circumference: 180, rotation: 270,
    }] },
    options: {
      cutout: "72%", maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    },
  });
  const bars = [["±10%", acc.within_10], ["±20%", acc.within_20], ["±30%", acc.within_30]];
  document.getElementById("accBars").innerHTML = bars
    .map(([lbl, val]) => `
      <div class="acc-bar"><span class="lbl">${lbl}</span>
        <span class="track"><span class="fill" style="width:${val}%"></span></span>
        <span class="pct">${val.toFixed(0)}%</span></div>`)
    .join("");
}

function renderCountryTable(rows) {
  const tb = document.querySelector("#countryTable tbody");
  const maxYield = Math.max(...rows.map((d) => d.avg_yield));
  tb.innerHTML = rows
    .map((d) => {
      const r2 = d.r2 === null ? "—" : d.r2.toFixed(3);
      const cls = d.r2 === null ? "mid" : d.r2 >= 0.8 ? "good" : "mid";
      const w = (d.avg_yield / maxYield) * 100;
      return `<tr><td>${d.country}</td>
        <td class="bar-cell">${fmt(d.avg_yield)}<span class="bar" style="width:${w}%"></span></td>
        <td><span class="pill ${cls}">${r2}</span></td><td>${d.n}</td><td></td></tr>`;
    })
    .join("");
}

/* ========================================================================== */
/*  Tester (uses the selected model)                                           */
/* ========================================================================== */
async function loadHistory() {
  const area = document.getElementById("selCountry").value;
  const item = document.getElementById("selCrop").value;
  const h = await getJSON(
    `/api/history?area=${encodeURIComponent(area)}&item=${encodeURIComponent(item)}`);
  if (h.available) {
    h.area = area; h.item = item;
    HISTORY = h;
    for (const [col, val] of Object.entries(h.values)) {
      const el = document.querySelector(`#advFields [data-col="${col}"]`);
      if (el) el.value = Number(val).toFixed(2);
    }
    manualDirty = false;
  }
  return h;
}

async function runPredict(area, item, year, overrides) {
  return getJSON("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ area, item, year, model: SELECTED, overrides }),
  });
}

async function estimate() {
  const btn = document.getElementById("btnPredict");
  const help = document.getElementById("testerHelp");
  const area = document.getElementById("selCountry").value;
  const item = document.getElementById("selCrop").value;
  btn.disabled = true;
  try {
    const h = await loadHistory();
    if (!h.available) {
      help.innerHTML = `<strong>Can't estimate:</strong> ${h.reason}`;
      document.getElementById("resultRow").hidden = true;
      return;
    }
    const res = await runPredict(area, item, h.year, {});
    help.innerHTML = `Ran <strong>${res.model}</strong> on
      <strong>${area} · ${item}</strong>'s ${h.year} record.`;
    showResult({ pred: res.prediction, actual: res.actual, year: res.year });
  } catch (e) {
    console.error(e);
    alert("Estimate failed — check the console.");
  } finally {
    btn.disabled = false;
    if (window.lucide) lucide.createIcons();
  }
}

async function estimateManual() {
  const btn = document.getElementById("btnPredictManual");
  const area = document.getElementById("selCountry").value;
  const item = document.getElementById("selCrop").value;
  btn.disabled = true;
  try {
    if (!HISTORY || HISTORY.area !== area || HISTORY.item !== item) {
      const h = await loadHistory();
      if (!h.available) { alert("No record to base this estimate on."); return; }
    }
    const overrides = {};
    document.querySelectorAll("#advFields [data-col]").forEach((el) => {
      overrides[el.dataset.col] = parseFloat(el.value);
    });
    const res = await runPredict(area, item, HISTORY.year, overrides);
    showResult({ pred: res.prediction, actual: manualDirty ? null : res.actual, year: res.year });
  } catch (e) {
    console.error(e);
    alert("Estimate failed — check the console.");
  } finally {
    btn.disabled = false;
    if (window.lucide) lucide.createIcons();
  }
}

function showResult({ pred, actual, year }) {
  document.getElementById("resultRow").hidden = false;
  document.getElementById("predVal").textContent = fmt(pred);
  document.getElementById("resYear").textContent = year ? `(${year})` : "";
  const actualCard = document.getElementById("actualCard");
  if (actual !== null && actual !== undefined) {
    const errPct = (Math.abs(pred - actual) / actual) * 100;
    const errEl = document.getElementById("errVal");
    actualCard.hidden = false;
    document.getElementById("actualVal").textContent = fmt(actual);
    errEl.textContent = `${errPct.toFixed(1)}% off`;
    errEl.className = "result-unit " + (errPct <= 20 ? "good" : "mid");
  } else {
    actualCard.hidden = true;
  }
}

/* ---- wire up -------------------------------------------------------------- */
document.getElementById("btnPredict").addEventListener("click", estimate);
document.getElementById("btnPredictManual").addEventListener("click", estimateManual);
document.getElementById("btnToggleAdv").addEventListener("click", async () => {
  const panel = document.getElementById("advancedPanel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden && !HISTORY) { try { await loadHistory(); } catch (_) {} }
  if (window.lucide) lucide.createIcons();
});
document.getElementById("advFields").addEventListener("input", () => { manualDirty = true; });
document.getElementById("selCountry").addEventListener("change", () => { HISTORY = null; });
document.getElementById("selCrop").addEventListener("change", () => { HISTORY = null; });

boot();
