"""
=============================================================================
CYP Dashboard — Clean Minimal Professional Results Viewer
=============================================================================
Serves a local web dashboard displaying all analysis results, plots,
and model evaluation metrics in a clean white/black design.

Usage:  python dashboard.py
Open:   http://localhost:8050
=============================================================================
"""

import os
import sys
import io
import json
import csv
import http.server
import webbrowser
import urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).parent
PORT = 8050


# ── Load evaluation data ─────────────────────────────────────────────────

def load_eval_csv():
    """Load evaluation_results.csv and return as list of dicts."""
    path = BASE_DIR / 'evaluation_output' / 'evaluation_results.csv'
    if not path.exists():
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Model'):
                rows.append(row)
    return rows


def load_eval_report():
    """Load the text evaluation report."""
    path = BASE_DIR / 'evaluation_output' / 'evaluation_report.txt'
    if path.exists():
        return path.read_text(encoding='utf-8')
    return 'Report not found. Run the pipeline first.'


# ── Build HTML ────────────────────────────────────────────────────────────

def build_metric_cards(results):
    """Build the metric highlight cards for the proposed model."""
    proposed = None
    best = None
    for r in results:
        if r['Model'] == '1D_CNN_Recursive_BiLSTM':
            proposed = r
        if best is None:
            best = r  # first row is the best (sorted by RMSE)

    if not proposed:
        proposed = best if best else {}

    cards_data = [
        ('R-Squared', f"{float(proposed.get('R²', 0)):.4f}", 'Variance Explained'),
        ('RMSE', f"{float(proposed.get('RMSE', 0)):.4f}", 'Root Mean Square Error'),
        ('MAE', f"{float(proposed.get('MAE', 0)):.4f}", 'Mean Absolute Error'),
        ('Accuracy', f"{float(proposed.get('Accuracy', 0))*100:.1f}%", 'Binned Classification'),
        ('F1-Score', f"{float(proposed.get('F1-Score', 0)):.4f}", 'Weighted Average'),
        ('MBE', f"{float(proposed.get('MBE', 0)):.4f}", 'Mean Bias Error'),
    ]

    html = ''
    for title, value, subtitle in cards_data:
        html += f'''
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>'''
    return html


def build_comparison_table(results):
    """Build the model comparison table."""
    if not results:
        return '<p>No results available.</p>'

    cols = ['Model', 'RMSE', 'MSE', 'MAE', 'MBE', 'R²', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    html = '<table><thead><tr>'
    for col in cols:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'

    for i, row in enumerate(results):
        row_class = ' class="highlight-row"' if row.get('Model') == '1D_CNN_Recursive_BiLSTM' else ''
        if i == 0:
            row_class = ' class="best-row"'
        html += f'<tr{row_class}>'
        for col in cols:
            val = row.get(col, '')
            if col != 'Model':
                try:
                    num = float(val)
                    if col == 'Accuracy':
                        val = f'{num*100:.2f}%'
                    else:
                        val = f'{num:.4f}'
                except (ValueError, TypeError):
                    pass
            else:
                val = val.replace('_', ' ')
            html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


def build_predictions_table():
    """Build the yield predictions sample table."""
    path = BASE_DIR / 'evaluation_output' / 'prediction_samples.csv'
    if not path.exists():
        return '<p>No prediction samples available. Run the evaluation module first.</p>'
        
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            rows.append(row)
            
    html = '<table><thead><tr>'
    for col in headers:
        display_col = col.replace('_', ' ')
        if display_col.startswith('Pred '):
            display_col = display_col[5:]
        html += f'<th>{display_col}</th>'
    html += '</tr></thead><tbody>'
    
    for row in rows:
        html += '<tr>'
        for i, val in enumerate(row):
            if i >= 3: # numeric columns
                try:
                    val = f'{float(val):,.1f}'
                except:
                    pass
            # Highlight the proposed model column
            cell_class = ' class="highlight-cell"' if headers[i] == 'Pred_1D_CNN_Recursive_BiLSTM' else ''
            html += f'<td{cell_class}>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


def collect_plot_paths():
    """Collect all plot paths organized by section."""
    sections = {
        'eda': {
            'title': 'Exploratory Data Analysis',
            'dir': BASE_DIR / 'analysis_output' / 'plots',
            'plots': [
                ('feature_distributions.png', 'Feature Distributions', 'Histograms of all four numerical features (yield, rainfall, pesticides, temperature) showing data spread, skewness, mean and median values.'),
                ('correlation_heatmap.png', 'Correlation Heatmap', 'Pearson correlation matrix between yield, rainfall, pesticides, and temperature features.'),
                ('boxplots_outliers.png', 'Outlier Detection', 'Box plots revealing the presence and count of statistical outliers in each feature using the IQR method.'),
                ('yield_by_crop.png', 'Yield by Crop Type', 'Horizontal bar chart comparing average yield (hg/ha) across all 10 crop types with standard deviation bars.'),
                ('yield_trends.png', 'Yield Trends Over Time', 'Time-series line plots showing yield evolution from 1990-2013 for the top 10 producing countries.'),
            ]
        },
        'feature_selection': {
            'title': 'Feature Selection',
            'dir': BASE_DIR / 'feature_selection_output' / 'plots',
            'plots': [
                ('lasso_feature_importance.png', 'LASSO Feature Importance', 'Absolute and signed LASSO regression coefficients ranking feature importance for yield prediction.'),
                ('sho_convergence.png', 'SHO Convergence', 'Selfish Herd Optimisation convergence curve showing fitness improvement over 30 iterations.'),
                ('feature_selection_comparison.png', 'Selection Comparison', 'Table comparing which features were selected by LASSO vs SHO meta-heuristic methods.'),
            ]
        },
        'training': {
            'title': 'Model Training',
            'dir': BASE_DIR / 'models' / 'plots',
            'plots': [
                ('training_history_1D_CNN.png', '1D CNN Training', 'Loss and MAE curves showing the 1D CNN training over 100 epochs with early stopping.'),
                ('training_history_Recursive_BiLSTM.png', 'BiLSTM Training', 'Loss and MAE curves for the standalone Recursive BiLSTM model (early stopped at epoch 26).'),
                ('training_history_1D_CNN_Recursive_BiLSTM.png', 'Hybrid Model Training', 'Training history for the proposed 1D CNN-Recursive BiLSTM hybrid architecture.'),
            ]
        },
        'evaluation': {
            'title': 'Model Evaluation',
            'dir': BASE_DIR / 'evaluation_output' / 'plots',
            'plots': [
                ('actual_vs_predicted.png', 'Actual vs Predicted', 'Scatter plots comparing true vs predicted yield for all 5 models. Closer to the red diagonal = better.'),
                ('metric_comparison.png', 'Metric Comparison', 'Bar charts comparing RMSE, MSE, MAE, R-squared, Accuracy, F1-Score, Precision and Recall across all models.'),
                ('residual_analysis.png', 'Residual Analysis', 'Residual scatter plots and error histograms showing prediction bias patterns for each model.'),
                ('error_distributions.png', 'Error Distributions', 'Overlaid absolute error histograms comparing error magnitude distributions across all models.'),
            ]
        },
    }
    return sections


def build_dashboard_html():
    """Build the complete dashboard HTML page."""
    results = load_eval_csv()
    sections = collect_plot_paths()

    # Find proposed model results
    proposed = {}
    for r in results:
        if r['Model'] == '1D_CNN_Recursive_BiLSTM':
            proposed = r
            break

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CYP — Crop Yield Prediction Results</title>
    <meta name="description" content="Crop Yield Prediction using 1D CNN-Recursive BiLSTM Hybrid Model — Complete Results Dashboard">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --white: #ffffff;
            --off-white: #fafafa;
            --light-gray: #f3f4f6;
            --border: #e5e7eb;
            --muted: #9ca3af;
            --text-secondary: #6b7280;
            --text: #111827;
            --black: #000000;
            --accent: #111827;
            --accent-light: #374151;
            --highlight: #f0fdf4;
            --highlight-border: #86efac;
            --proposed-bg: #eff6ff;
            --proposed-border: #93c5fd;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--white);
            color: var(--text);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}

        /* ── Navigation ── */
        nav {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 0 2rem;
        }}
        nav .nav-inner {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 56px;
        }}
        nav .logo {{
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--black);
        }}
        nav .logo span {{
            font-weight: 400;
            color: var(--muted);
            margin-left: 6px;
        }}
        nav .nav-links {{
            display: flex;
            gap: 2rem;
            list-style: none;
        }}
        nav .nav-links a {{
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-decoration: none;
            letter-spacing: 0.02em;
            transition: color 0.2s;
        }}
        nav .nav-links a:hover {{
            color: var(--black);
        }}

        /* ── Sections ── */
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }}

        .hero {{
            padding: 4rem 0 3rem;
            border-bottom: 1px solid var(--border);
        }}
        .hero h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.2;
            margin-bottom: 0.75rem;
        }}
        .hero .subtitle {{
            font-size: 1.05rem;
            color: var(--text-secondary);
            font-weight: 400;
            max-width: 680px;
            line-height: 1.7;
        }}
        .hero .tags {{
            display: flex;
            gap: 0.5rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }}
        .hero .tag {{
            font-size: 0.72rem;
            font-weight: 500;
            padding: 0.3rem 0.75rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--text-secondary);
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}

        section {{
            padding: 3.5rem 0;
            border-bottom: 1px solid var(--border);
        }}
        section:last-child {{
            border-bottom: none;
        }}
        .section-header {{
            margin-bottom: 2rem;
        }}
        .section-header h2 {{
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.4rem;
        }}
        .section-header p {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            max-width: 600px;
        }}
        .section-number {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.4rem;
        }}

        /* ── Metric Cards ── */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 1px;
            background: var(--border);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 2.5rem;
        }}
        .metric-card {{
            background: var(--white);
            padding: 1.5rem;
            text-align: center;
        }}
        .metric-label {{
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.5rem;
        }}
        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--black);
            line-height: 1;
            margin-bottom: 0.3rem;
        }}
        .metric-subtitle {{
            font-size: 0.68rem;
            color: var(--muted);
        }}

        /* ── Table ── */
        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }}
        thead {{
            background: var(--off-white);
        }}
        th {{
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--light-gray);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            white-space: nowrap;
        }}
        td:first-child {{
            font-family: 'Inter', sans-serif;
            font-weight: 500;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .best-row {{
            background: var(--highlight);
        }}
        .best-row td:first-child::after {{
            content: ' \\2713';
            color: #22c55e;
            font-weight: 700;
            margin-left: 4px;
        }}
        .highlight-row {{
            background: var(--proposed-bg);
        }}
        .highlight-row td:first-child::after {{
            content: ' (proposed)';
            color: #3b82f6;
            font-size: 0.65rem;
            font-weight: 500;
        }}
        .highlight-cell {{
            background: var(--proposed-bg);
            color: #1e40af;
            font-weight: 600;
        }}

        /* ── Plots ── */
        .plot-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }}
        .plot-card {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        .plot-card .plot-header {{
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border);
            background: var(--off-white);
        }}
        .plot-card .plot-header h3 {{
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }}
        .plot-card .plot-header p {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }}
        .plot-card img {{
            width: 100%;
            display: block;
            background: var(--white);
        }}
        .plot-grid-2 {{
            grid-template-columns: repeat(2, 1fr);
        }}

        /* ── Pipeline ── */
        .pipeline {{
            display: flex;
            align-items: center;
            gap: 0;
            margin: 2rem 0;
            overflow-x: auto;
            padding-bottom: 0.5rem;
        }}
        .pipeline-step {{
            flex-shrink: 0;
            padding: 1rem 1.5rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            text-align: center;
            min-width: 160px;
            background: var(--white);
        }}
        .pipeline-step.active {{
            background: var(--black);
            color: var(--white);
            border-color: var(--black);
        }}
        .pipeline-step .step-num {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: var(--muted);
            margin-bottom: 0.3rem;
        }}
        .pipeline-step.active .step-num {{
            color: rgba(255,255,255,0.5);
        }}
        .pipeline-step .step-name {{
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .pipeline-arrow {{
            flex-shrink: 0;
            width: 40px;
            text-align: center;
            color: var(--muted);
            font-size: 1.2rem;
        }}

        /* ── Dataset summary ── */
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        .stat-item {{
            padding: 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
        }}
        .stat-item .stat-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--muted);
            font-weight: 500;
        }}
        .stat-item .stat-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }}

        /* ── Footer ── */
        footer {{
            padding: 2.5rem 0;
            text-align: center;
            color: var(--muted);
            font-size: 0.75rem;
            border-top: 1px solid var(--border);
        }}

        /* ── Responsive ── */
        @media (max-width: 768px) {{
            .plot-grid-2 {{
                grid-template-columns: 1fr;
            }}
            nav .nav-links {{
                gap: 1rem;
            }}
            .hero h1 {{
                font-size: 1.6rem;
            }}
            .pipeline {{
                flex-direction: column;
            }}
            .pipeline-arrow {{
                transform: rotate(90deg);
            }}
        }}
    </style>
</head>
<body>

<nav>
    <div class="nav-inner">
        <div class="logo">CYP<span>Dashboard</span></div>
        <ul class="nav-links">
            <li><a href="#overview">Overview</a></li>
            <li><a href="#eda">EDA</a></li>
            <li><a href="#features">Features</a></li>
            <li><a href="#training">Training</a></li>
            <li><a href="#evaluation">Evaluation</a></li>
            <li><a href="#predictions">Predictions</a></li>
        </ul>
    </div>
</nav>

<div class="container">

    <!-- HERO -->
    <div class="hero" id="overview">
        <h1>Crop Yield Prediction</h1>
        <p class="subtitle">
            A hybrid deep learning approach using 1D CNN-Recursive BiLSTM
            for predicting agricultural crop yields from climate, pesticide,
            and geographical data across 101 countries and 10 crop types.
        </p>
        <div class="tags">
            <span class="tag">Deep Learning</span>
            <span class="tag">1D CNN</span>
            <span class="tag">BiLSTM</span>
            <span class="tag">LASSO</span>
            <span class="tag">SHO</span>
            <span class="tag">Crop Yield</span>
            <span class="tag">TensorFlow</span>
        </div>
    </div>

    <!-- PIPELINE -->
    <section>
        <div class="section-number">Architecture</div>
        <div class="section-header">
            <h2>Pipeline Overview</h2>
            <p>End-to-end workflow from raw agricultural data to yield prediction.</p>
        </div>
        <div class="pipeline">
            <div class="pipeline-step">
                <div class="step-num">01</div>
                <div class="step-name">Data Analysis</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step">
                <div class="step-num">02</div>
                <div class="step-name">Preprocessing</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step">
                <div class="step-num">03</div>
                <div class="step-name">Feature Selection</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step active">
                <div class="step-num">04</div>
                <div class="step-name">CNN-BiLSTM</div>
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step">
                <div class="step-num">05</div>
                <div class="step-name">Evaluation</div>
            </div>
        </div>

        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-label">Total Samples</div>
                <div class="stat-value">25,932</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Countries</div>
                <div class="stat-value">101</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Crop Types</div>
                <div class="stat-value">10</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Year Range</div>
                <div class="stat-value">1990-2013</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Features</div>
                <div class="stat-value">6</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Models Trained</div>
                <div class="stat-value">5</div>
            </div>
        </div>
    </section>

    <!-- PROPOSED MODEL METRICS -->
    <section>
        <div class="section-number">Proposed Model</div>
        <div class="section-header">
            <h2>1D CNN-Recursive BiLSTM — Key Metrics</h2>
            <p>Performance of the proposed hybrid architecture on the held-out test set (5,187 samples).</p>
        </div>
        <div class="metrics-grid">
            {build_metric_cards(results)}
        </div>
    </section>

    <!-- COMPARISON TABLE -->
    <section>
        <div class="section-number">Comparison</div>
        <div class="section-header">
            <h2>All Models — Performance Table</h2>
            <p>Sorted by RMSE (lower is better). Green row = best overall. Blue row = proposed model.</p>
        </div>
        <div class="table-wrap">
            {build_comparison_table(results)}
        </div>
    </section>

    <!-- PREDICTIONS TABLE -->
    <section id="predictions">
        <div class="section-number">Results</div>
        <div class="section-header">
            <h2>Predict the Yield</h2>
            <p>Randomly sampled test set predictions comparing actual vs predicted crop yields across all models. Values are in hectograms per hectare (hg/ha). The proposed model predictions are highlighted.</p>
        </div>
        <div class="table-wrap" style="max-height: 600px; overflow-y: auto;">
            {build_predictions_table()}
        </div>
    </section>
'''

    # Build plot sections
    section_ids = {'eda': 'eda', 'feature_selection': 'features', 
                   'training': 'training', 'evaluation': 'evaluation'}

    for key, section in sections.items():
        sec_id = section_ids.get(key, key)
        step_num = {'eda': '01', 'feature_selection': '03', 
                    'training': '04', 'evaluation': '05'}.get(key, '')

        html += f'''
    <section id="{sec_id}">
        <div class="section-number">Stage {step_num}</div>
        <div class="section-header">
            <h2>{section["title"]}</h2>
        </div>
        <div class="plot-grid">'''

        for filename, title, description in section['plots']:
            filepath = section['dir'] / filename
            if filepath.exists():
                rel = filepath.relative_to(BASE_DIR).as_posix()
                html += f'''
            <div class="plot-card">
                <div class="plot-header">
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                <img src="/plots/{rel}" alt="{title}" loading="lazy" />
            </div>'''

        html += '''
        </div>
    </section>'''

    html += '''

    <footer>
        Crop Yield Prediction &mdash; 1D CNN-Recursive BiLSTM Hybrid Model &mdash; CYP Pipeline v1.0
    </footer>

</div>

<script>
    // Smooth scroll for nav links
    document.querySelectorAll('nav a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            e.preventDefault();
            const target = document.querySelector(a.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
</script>

</body>
</html>'''

    return html


# ── HTTP Server ───────────────────────────────────────────────────────────

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Serve the dashboard and plot images."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/index.html':
            html = build_dashboard_html()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        elif path.startswith('/plots/'):
            # Serve plot images
            rel_path = path[7:]  # Remove /plots/
            file_path = BASE_DIR / rel_path
            if file_path.exists() and file_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.svg'):
                self.send_response(200)
                ct = 'image/png' if file_path.suffix == '.png' else 'image/jpeg'
                self.send_header('Content-Type', ct)
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(file_path.read_bytes())
            else:
                self.send_error(404)

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Quieter logging
        pass


def main():
    server = http.server.HTTPServer(('localhost', PORT), DashboardHandler)
    url = f'http://localhost:{PORT}'
    print(f'\n  CYP Dashboard running at: {url}')
    print(f'  Press Ctrl+C to stop.\n')
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  Dashboard stopped.')
        server.server_close()


if __name__ == '__main__':
    main()
