/* ============================================================================
   BiLSTM Crop-Yield dashboard — frontend logic  (light theme)
   ========================================================================== */
const FEATURE_COLS = [
  "hg/ha_yield",
  "average_rain_fall_mm_per_year",
  "pesticides_tonnes",
  "avg_temp",
];
let META = null;
let HISTORY = null;       // last autofilled window {window, actual, target_year, area, item}
let manualDirty = false;  // true once the user edits the manual table

/* ---- chart.js light theme ------------------------------------------------- */
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

/* ========================================================================== */
/*  Boot                                                                       */
/* ========================================================================== */
async function boot() {
  try {
    META = await getJSON("/api/meta");
    renderKpis(META);
    populateSelects(META);
    buildInputRows();
    const findings = await getJSON("/api/findings");
    renderCharts(findings);
    renderAccuracy(findings.accuracy);
    renderCountryTable(findings.per_country);
  } catch (e) {
    console.error(e);
  } finally {
    if (window.lucide) lucide.createIcons();
  }
}

/* ---- KPIs ----------------------------------------------------------------- */
function renderKpis(m) {
  const t = m.metrics.test;
  document.getElementById("kpiR2").textContent = t.R2.toFixed(3);
  document.getElementById("kpiRmse").textContent = fmt(t.RMSE);
  document.getElementById("kpiSeq").textContent = fmt(m.n_sequences);
  document.getElementById("specSeq").textContent =
    `${fmt(m.n_sequences)} windows (101 countries × 10 crops)`;
  document.getElementById("cmpR2").textContent = t.R2.toFixed(3);
}

/* ---- selects -------------------------------------------------------------- */
function populateSelects(m) {
  const c = document.getElementById("selCountry");
  const k = document.getElementById("selCrop");
  m.countries.forEach((x) => c.add(new Option(x, x)));
  m.crops.forEach((x) => k.add(new Option(x, x)));
  if (m.countries.includes("India")) c.value = "India";
  if (m.crops.includes("Maize")) k.value = "Maize";
}

/* ---- editable input table (for advanced mode) ----------------------------- */
function buildInputRows() {
  const tb = document.querySelector("#inputTable tbody");
  tb.innerHTML = "";
  const means = META.feature_stats;
  for (let i = 0; i < META.window; i++) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td><input data-role="year" type="number" value="${1999 + i}" /></td>` +
      FEATURE_COLS.map(
        (f) =>
          `<td><input data-feature="${f}" type="number" step="any" value="${means[f].mean.toFixed(
            2
          )}" /></td>`
      ).join("");
    tb.appendChild(tr);
  }
}

function fillTable(windowRows) {
  const rows = document.querySelectorAll("#inputTable tbody tr");
  windowRows.forEach((wr, i) => {
    const tr = rows[i];
    if (!tr) return;
    tr.querySelector('[data-role="year"]').value = wr.year;
    FEATURE_COLS.forEach((f) => {
      tr.querySelector(`[data-feature="${f}"]`).value = Number(wr[f]).toFixed(2);
    });
  });
  manualDirty = false;
}

function readTable() {
  return [...document.querySelectorAll("#inputTable tbody tr")].map((tr) => {
    const o = {};
    FEATURE_COLS.forEach((f) => {
      o[f] = parseFloat(tr.querySelector(`[data-feature="${f}"]`).value) || 0;
    });
    return o;
  });
}

/* ========================================================================== */
/*  Charts                                                                     */
/* ========================================================================== */
function renderCharts(f) {
  new Chart(document.getElementById("scatterChart"), {
    type: "scatter",
    data: {
      datasets: [
        { label: "Test sequences", data: f.scatter, pointRadius: 2.5,
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

  new Chart(document.getElementById("trendChart"), {
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
  new Chart(document.getElementById("cropChart"), {
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
  new Chart(document.getElementById("resChart"), {
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

/* ---- accuracy gauge + bars ------------------------------------------------ */
function renderAccuracy(acc) {
  // headline KPI
  document.getElementById("kpiAcc").textContent = acc.within_20.toFixed(1) + "%";
  document.getElementById("gaugeVal").textContent = acc.within_20.toFixed(0) + "%";
  document.getElementById("cmpAcc").textContent = acc.within_20.toFixed(1) + "%";

  // half-doughnut gauge
  const v = acc.within_20;
  new Chart(document.getElementById("accGauge"), {
    type: "doughnut",
    data: {
      datasets: [{
        data: [v, 100 - v],
        backgroundColor: ["rgba(13,148,136,0.9)", "rgba(15,23,42,0.07)"],
        borderWidth: 0, circumference: 180, rotation: 270,
      }],
    },
    options: {
      cutout: "72%", maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    },
  });

  // tolerance bars
  const bars = [
    ["±10%", acc.within_10],
    ["±20%", acc.within_20],
    ["±30%", acc.within_30],
  ];
  document.getElementById("accBars").innerHTML = bars
    .map(([lbl, val]) => `
      <div class="acc-bar">
        <span class="lbl">${lbl}</span>
        <span class="track"><span class="fill" style="width:${val}%"></span></span>
        <span class="pct">${val.toFixed(0)}%</span>
      </div>`)
    .join("");
}

/* ---- country table -------------------------------------------------------- */
function renderCountryTable(rows) {
  const tb = document.querySelector("#countryTable tbody");
  const maxYield = Math.max(...rows.map((d) => d.avg_yield));
  tb.innerHTML = rows
    .map((d) => {
      const r2 = d.r2 === null ? "—" : d.r2.toFixed(3);
      const cls = d.r2 === null ? "mid" : d.r2 >= 0.8 ? "good" : "mid";
      const w = (d.avg_yield / maxYield) * 100;
      return `<tr>
        <td>${d.country}</td>
        <td class="bar-cell">${fmt(d.avg_yield)}<span class="bar" style="width:${w}%"></span></td>
        <td><span class="pill ${cls}">${r2}</span></td>
        <td>${d.n}</td><td></td>
      </tr>`;
    })
    .join("");
}

/* ========================================================================== */
/*  Tester                                                                     */
/* ========================================================================== */
async function loadHistory() {
  const area = document.getElementById("selCountry").value;
  const item = document.getElementById("selCrop").value;
  const h = await getJSON(
    `/api/history?area=${encodeURIComponent(area)}&item=${encodeURIComponent(item)}`
  );
  if (h.available) {
    h.area = area; h.item = item;
    HISTORY = h;
    fillTable(h.window);
  }
  return h;
}

async function runPredict(windowRows, area, item) {
  const res = await getJSON("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ area, item, window: windowRows }),
  });
  return res.prediction;
}

/* Primary one-click flow: load real history -> forecast -> compare to actual */
async function forecast() {
  const btn = document.getElementById("btnPredict");
  const help = document.getElementById("testerHelp");
  const area = document.getElementById("selCountry").value;
  const item = document.getElementById("selCrop").value;
  btn.disabled = true;
  try {
    const h = await loadHistory();
    if (!h.available) {
      help.innerHTML = `<strong>Can't forecast:</strong> ${h.reason}
        Try another country/crop, or use manual mode.`;
      document.getElementById("resultRow").hidden = true;
      return;
    }
    const pred = await runPredict(h.window, area, item);
    help.innerHTML = `Used <strong>${area} · ${item}</strong>'s real history
      (${h.window[0].year}–${h.window[h.window.length - 1].year}) to forecast
      <strong>${h.target_year}</strong>.`;
    showResult({ pred, actual: h.actual, targetYear: h.target_year });
  } catch (e) {
    console.error(e);
    alert("Forecast failed — check the console.");
  } finally {
    btn.disabled = false;
    if (window.lucide) lucide.createIcons();
  }
}

/* Advanced flow: forecast from whatever is in the table right now */
async function forecastManual() {
  const btn = document.getElementById("btnPredictManual");
  const area = document.getElementById("selCountry").value;
  const item = document.getElementById("selCrop").value;
  btn.disabled = true;
  try {
    const windowRows = readTable();
    const pred = await runPredict(windowRows, area, item);
    // show held-out actual only if the values still match the autofilled history
    const matched = HISTORY && !manualDirty &&
      HISTORY.area === area && HISTORY.item === item;
    showResult({
      pred,
      actual: matched ? HISTORY.actual : null,
      targetYear: matched ? HISTORY.target_year : null,
    });
  } catch (e) {
    console.error(e);
    alert("Forecast failed — check the console.");
  } finally {
    btn.disabled = false;
    if (window.lucide) lucide.createIcons();
  }
}

function showResult({ pred, actual, targetYear }) {
  document.getElementById("resultRow").hidden = false;
  document.getElementById("predVal").textContent = fmt(pred);
  document.getElementById("resYear").textContent = targetYear ? `(${targetYear})` : "";

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
document.getElementById("btnPredict").addEventListener("click", forecast);
document.getElementById("btnPredictManual").addEventListener("click", forecastManual);

document.getElementById("btnToggleAdv").addEventListener("click", async (e) => {
  const panel = document.getElementById("advancedPanel");
  panel.hidden = !panel.hidden;
  // populate with real history the first time it's opened
  if (!panel.hidden && !HISTORY) {
    try { await loadHistory(); } catch (_) {}
  }
  if (window.lucide) lucide.createIcons();
});

document.getElementById("inputTable").addEventListener("input", () => { manualDirty = true; });
document.getElementById("selCountry").addEventListener("change", () => { HISTORY = null; });
document.getElementById("selCrop").addEventListener("change", () => { HISTORY = null; });

boot();
