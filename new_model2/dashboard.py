"""
=============================================================================
Interactive Dash Dashboard for Old Dataset Crop Yield Prediction (new_model2)
=============================================================================
Provides an interactive web dashboard with:
  - Dataset statistics and distribution
  - Model comparison table (all 8 models)
  - Actual vs Predicted plots
  - Per-crop and per-country analysis
  - RMSE & R² comparison charts
=============================================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import dash
    from dash import dcc, html, dash_table, Input, Output, State
    import dash_bootstrap_components as dbc
except ImportError:
    print("  Installing Dash and Bootstrap Components...")
    os.system('pip3 install dash dash-bootstrap-components')
    import dash
    from dash import dcc, html, dash_table, Input, Output, State
    import dash_bootstrap_components as dbc

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'preprocessed_data')
MODEL_DIR        = os.path.join(BASE_DIR, 'models')
EVAL_DIR         = os.path.join(BASE_DIR, 'evaluation_output')


def load_data():
    """Load all required data for the dashboard."""
    data = {}

    results_path = os.path.join(EVAL_DIR, 'evaluation_results.csv')
    if os.path.exists(results_path):
        data['results'] = pd.read_csv(results_path)

    samples_path = os.path.join(EVAL_DIR, 'prediction_samples.csv')
    if os.path.exists(samples_path):
        data['samples'] = pd.read_csv(samples_path)

    preprocessed_path = os.path.join(PREPROCESSED_DIR, 'preprocessed_data.csv')
    if os.path.exists(preprocessed_path):
        data['preprocessed'] = pd.read_csv(preprocessed_path)

    enc_path = os.path.join(PREPROCESSED_DIR, 'encoders.pkl')
    if os.path.exists(enc_path):
        with open(enc_path, 'rb') as f:
            data['encoders'] = pickle.load(f)

    scaler_path = os.path.join(PREPROCESSED_DIR, 'scalers.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            data['scalers'] = pickle.load(f)

    split_path = os.path.join(PREPROCESSED_DIR, 'train_test_splits.npz')
    if os.path.exists(split_path):
        splits = np.load(split_path)
        data['y_test_raw'] = splits['y_test_raw']
        data['X_test']     = splits['X_test']

    opt_a_path = os.path.join(MODEL_DIR, 'option_a_predictions.npz')
    if os.path.exists(opt_a_path):
        pred_a = np.load(opt_a_path)
        data['opt_a_preds'] = {k: pred_a[k] for k in pred_a.files}

    lstm_path = os.path.join(MODEL_DIR, 'lstm_predictions.npz')
    if os.path.exists(lstm_path):
        pred_lstm = np.load(lstm_path)
        data['lstm_preds'] = {k: pred_lstm[k] for k in pred_lstm.files}

    return data


# ── Application Setup ──────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True)
app.title = "CYP — Old Dataset Crop Yield Prediction Dashboard"

# ── Colour Palette ──────────────────────────────────────────────────────────
COLORS = {
    'primary': '#6366f1',
    'success': '#22c55e',
    'warning': '#f59e0b',
    'danger':  '#ef4444',
    'info':    '#3b82f6',
    'bg':      '#0f172a',
    'card':    '#1e293b',
    'text':    '#e2e8f0',
    'muted':   '#94a3b8',
}

OPTION_COLORS = {
    'A': '#22c55e',
    'B': '#6366f1',
    'C': '#f59e0b',
}


def build_metric_card(title, value, subtitle, color):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H3(value, style={"color": color, "fontWeight": "bold"}),
            html.P(subtitle, className="text-muted mb-0", style={"fontSize": "0.78rem"}),
        ])
    ], style={"background": COLORS['card'], "border": f"1px solid {color}33"})


def build_layout(data):
    """Build the main layout, handling missing data gracefully."""
    has_results      = 'results' in data and data['results'] is not None
    has_preprocessed = 'preprocessed' in data
    has_samples      = 'samples' in data

    results  = data.get('results', pd.DataFrame())
    pre_df   = data.get('preprocessed', pd.DataFrame())
    samples  = data.get('samples', pd.DataFrame())

    # ── Overview Metrics ──
    total_samples = len(pre_df) if has_preprocessed else "N/A"
    n_crops       = pre_df['Item'].nunique() if has_preprocessed and 'Item' in pre_df.columns else "N/A"
    n_countries   = pre_df['Area'].nunique() if has_preprocessed and 'Area' in pre_df.columns else "N/A"
    year_range    = (f"{int(pre_df['Year'].min())} – {int(pre_df['Year'].max())}"
                     if has_preprocessed else "N/A")

    best_rmse  = f"{results.iloc[0]['RMSE']:.1f} hg/ha" if has_results else "Run Pipeline First"
    best_r2    = f"{results.iloc[0]['R²']:.4f}"         if has_results else "—"
    best_model = results.iloc[0]['Model']                if has_results else "—"

    if has_results:
        display_results = results[['Model', 'RMSE', 'MSE', 'MAE', 'MBE', 'R²', 'Acc_10%', 'Acc_20%']].round(3)
        display_results = display_results.rename(columns={
            'Acc_10%': 'Accuracy (±10%)',
            'Acc_20%': 'Accuracy (±20%)'
        })
        leaderboard_table = dash_table.DataTable(
            data=display_results.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in display_results.columns],
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': COLORS['primary'], 'color': 'white',
                          'fontWeight': 'bold', 'textAlign': 'center'},
            style_data={'backgroundColor': COLORS['card'], 'color': COLORS['text'],
                        'border': '1px solid #334155'},
            style_data_conditional=[
                {'if': {'row_index': 0},
                 'backgroundColor': '#14532d',
                 'fontWeight': 'bold'}
            ],
            style_cell={'textAlign': 'center', 'padding': '8px'},
            sort_action='native',
        )
    else:
        leaderboard_table = html.P(
            "No evaluation results found. Run `python3 main.py` to generate results.",
            className="text-warning text-center mt-3"
        )

    # ── Main Yield Distribution Plot ──
    if has_preprocessed and 'hg/ha_yield' in pre_df.columns:
        yield_hist = px.histogram(
            pre_df.sample(min(20000, len(pre_df)), random_state=42),
            x='hg/ha_yield',
            color='Item' if 'Item' in pre_df.columns else None,
            title="Yield Distribution Across Crops",
            labels={'hg/ha_yield': 'Yield (hg/ha)'},
            template='plotly_dark'
        )
        yield_hist.update_layout(paper_bgcolor=COLORS['card'], plot_bgcolor='rgba(0,0,0,0)',
                                  font_color=COLORS['text'])
    else:
        yield_hist = go.Figure().update_layout(title="Yield distribution not available",
                                               paper_bgcolor=COLORS['card'])

    # ── Average Yield by Crop Bar Chart ──
    if has_preprocessed and 'Item' in pre_df.columns:
        crop_stats = (pre_df.groupby('Item')['hg/ha_yield']
                            .mean()
                            .sort_values(ascending=False)
                            .reset_index())
        crop_bar = px.bar(crop_stats, x='Item', y='hg/ha_yield',
                          title="Average Yield by Crop",
                          labels={'hg/ha_yield': 'Avg Yield (hg/ha)', 'Item': 'Crop'},
                          color='hg/ha_yield', color_continuous_scale='Viridis',
                          template='plotly_dark')
        crop_bar.update_layout(paper_bgcolor=COLORS['card'], plot_bgcolor='rgba(0,0,0,0)',
                                font_color=COLORS['text'], xaxis_tickangle=-35)
    else:
        crop_bar = go.Figure().update_layout(title="Crop analysis not available",
                                             paper_bgcolor=COLORS['card'])

    # ── Top 10 Countries by Average Yield ──
    if has_preprocessed and 'Area' in pre_df.columns:
        country_stats = (pre_df.groupby('Area')['hg/ha_yield']
                               .mean()
                               .sort_values(ascending=False)
                               .head(10)
                               .reset_index())
        country_bar = px.bar(country_stats, x='Area', y='hg/ha_yield',
                             title="Top 10 Countries by Average Yield",
                             labels={'hg/ha_yield': 'Avg Yield (hg/ha)', 'Area': 'Country'},
                             color='hg/ha_yield', color_continuous_scale='Blues',
                             template='plotly_dark')
        country_bar.update_layout(paper_bgcolor=COLORS['card'], plot_bgcolor='rgba(0,0,0,0)',
                                   font_color=COLORS['text'], xaxis_tickangle=-35)
    else:
        country_bar = go.Figure().update_layout(title="Country analysis not available",
                                                paper_bgcolor=COLORS['card'])

    # ── Model RMSE & R² Comparison ──
    if has_results:
        sorted_r = results.sort_values('RMSE')
        option_colors = [OPTION_COLORS.get(m.split(':')[0].replace('Option ', '').strip(), COLORS['primary'])
                         for m in sorted_r['Model']]
        rmse_bar = go.Figure([go.Bar(
            x=sorted_r['Model'], y=sorted_r['RMSE'],
            marker_color=option_colors, text=sorted_r['RMSE'].round(1), textposition='outside'
        )])
        rmse_bar.update_layout(
            title="RMSE Comparison — All Models (hg/ha, Lower is Better)",
            xaxis_tickangle=-25, template='plotly_dark',
            paper_bgcolor=COLORS['card'], plot_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text']
        )

        r2_bar = go.Figure([go.Bar(
            x=sorted_r.sort_values('R²', ascending=False)['Model'],
            y=sorted_r.sort_values('R²', ascending=False)['R²'],
            marker_color=option_colors[::-1],
            text=sorted_r.sort_values('R²', ascending=False)['R²'].round(4),
            textposition='outside'
        )])
        r2_bar.update_layout(
            title="R² Score — All Models (Higher is Better)",
            xaxis_tickangle=-25, template='plotly_dark',
            paper_bgcolor=COLORS['card'], plot_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'], yaxis_range=[0, 1.05]
        )
    else:
        rmse_bar = go.Figure().update_layout(title="Run pipeline to see RMSE comparison",
                                              paper_bgcolor=COLORS['card'])
        r2_bar   = go.Figure().update_layout(title="Run pipeline to see R² comparison",
                                              paper_bgcolor=COLORS['card'])

    # ── Prediction Samples Table ──
    if has_samples:
        pred_table = dash_table.DataTable(
            data=samples.head(20).to_dict('records'),
            columns=[{'name': c, 'id': c} for c in samples.columns],
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': COLORS['primary'], 'color': 'white',
                          'fontWeight': 'bold', 'textAlign': 'center'},
            style_data={'backgroundColor': COLORS['card'], 'color': COLORS['text'],
                        'border': '1px solid #334155'},
            style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '0.8rem'},
            page_size=10,
        )
    else:
        pred_table = html.P("Prediction samples not available. Run the pipeline first.",
                            className="text-warning text-center mt-3")

    return dbc.Container([
        # ── Header ──
        dbc.Row([
            dbc.Col([
                html.H1("🌾 Crop Yield Prediction — Old Dataset", className="text-white fw-bold mb-1"),
                html.P(f"101 Countries · 10 Crops · {year_range} · Options A / B / C Hybrid Models",
                       className="text-muted"),
            ], width=12)
        ], className="my-4"),

        # ── Summary Metrics ──
        dbc.Row([
            dbc.Col(build_metric_card("Total Samples",
                                      f"{total_samples:,}" if isinstance(total_samples, int) else total_samples,
                                      "Lagged rows after preprocessing", COLORS['success']), md=3),
            dbc.Col(build_metric_card("Unique Crops", str(n_crops), "Crop categories in dataset", COLORS['info']), md=3),
            dbc.Col(build_metric_card("Countries", str(n_countries), "Countries in dataset", COLORS['warning']), md=3),
            dbc.Col(build_metric_card("Best Model RMSE", best_rmse,
                                      best_model.split(': ')[-1] if ':' in best_model else best_model,
                                      COLORS['primary']), md=3),
        ], className="mb-4 g-3"),

        # ── Leaderboard ──
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 Model Leaderboard (Sorted by RMSE — hg/ha)", className="mb-0 text-white")),
                    dbc.CardBody(leaderboard_table)
                ], style={"background": COLORS['card']})
            ], width=12)
        ], className="mb-4"),

        # ── RMSE & R² Comparison Charts ──
        dbc.Row([
            dbc.Col([
                dbc.Card([dbc.CardBody(dcc.Graph(figure=rmse_bar))],
                         style={"background": COLORS['card']})
            ], md=6),
            dbc.Col([
                dbc.Card([dbc.CardBody(dcc.Graph(figure=r2_bar))],
                         style={"background": COLORS['card']})
            ], md=6),
        ], className="mb-4"),

        # ── Yield Distribution & Crop/Country Analysis ──
        dbc.Row([
            dbc.Col([
                dbc.Card([dbc.CardBody(dcc.Graph(figure=yield_hist))],
                         style={"background": COLORS['card']})
            ], md=7),
            dbc.Col([
                dbc.Card([dbc.CardBody(dcc.Graph(figure=crop_bar))],
                         style={"background": COLORS['card']})
            ], md=5),
        ], className="mb-4"),

        # ── Country analysis ──
        dbc.Row([
            dbc.Col([
                dbc.Card([dbc.CardBody(dcc.Graph(figure=country_bar))],
                         style={"background": COLORS['card']})
            ], width=12)
        ], className="mb-4"),

        # ── Prediction Samples ──
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🔍 Prediction Samples (Test Set Preview)", className="mb-0 text-white")),
                    dbc.CardBody(pred_table)
                ], style={"background": COLORS['card']})
            ], width=12)
        ], className="mb-4"),

    ], fluid=True, style={"backgroundColor": COLORS['bg'], "minHeight": "100vh"})


# ── Bootstrap initial layout ──
data = load_data()
app.layout = build_layout(data)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  CROP YIELD PREDICTION DASHBOARD (OLD DATASET)")
    print("  Dashboard URL: http://localhost:8050")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8050)
