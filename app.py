"""
RUL Prediction Dashboard — Proof of Concept
Remaining Useful Life prediction for industrial coolant pump assemblies.
Bayesian hierarchical model with interactive parameter exploration.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gamma, weibull_min

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Prediction — Coolant Pump Fleet",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme Colors ─────────────────────────────────────────────────────
NAVY = "#1B3A5C"
ACCENT = "#2E75B6"
LIGHT_BLUE = "#A8D0E6"
GREEN = "#27AE60"
AMBER = "#F39C12"
RED = "#E74C3C"
DARK_GRAY = "#2C3E50"
MED_GRAY = "#7F8C8D"
BG = "#FAFBFC"

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    code, .stCode, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1B3A5C;
        font-weight: 700;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border-left: 4px solid;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        margin: 0 0 0.25rem 0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #7F8C8D;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 0.78rem;
        color: #7F8C8D;
        margin-top: 0.2rem;
    }
    .healthy { border-color: #27AE60; }
    .healthy .value { color: #27AE60; }
    .watch { border-color: #F39C12; }
    .watch .value { color: #F39C12; }
    .critical { border-color: #E74C3C; }
    .critical .value { color: #E74C3C; }
    .offline { border-color: #7F8C8D; }
    .offline .value { color: #7F8C8D; }

    .stSidebar [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1B3A5C 0%, #0f2640 100%);
        color: white;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar label, .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown {
        color: white !important;
    }
    .stSidebar [data-testid="stSidebarContent"] hr {
        border-color: rgba(255,255,255,0.15);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
    .plot-container { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ RUL Model Parameters")
    st.markdown("Adjust the Bayesian model inputs to explore how they affect predictions.")
    st.markdown("---")

    st.markdown("### Degradation Model")
    deg_rate = st.slider("Degradation Rate", 0.1, 1.0, 0.40, 0.05,
                         help="Average monthly health index decline")
    deg_accel = st.slider("Degradation Acceleration", 0.001, 0.015, 0.005, 0.001,
                          format="%.3f",
                          help="Quadratic acceleration of degradation over time")
    noise_std = st.slider("Process Noise (σ)", 0.1, 1.0, 0.35, 0.05,
                          help="Stochastic variability in degradation path")

    st.markdown("---")
    st.markdown("### Failure Threshold")
    failure_threshold = st.slider("Health Index Threshold", 20, 60, 40, 5,
                                  help="Health index below which component is considered failed")

    st.markdown("---")
    st.markdown("### Fleet Parameters")
    weibull_shape = st.slider("Weibull Shape (β)", 1.5, 6.0, 3.5, 0.1,
                              help="Population failure distribution shape. Higher = more concentrated failures")
    weibull_scale = st.slider("Weibull Scale (η, months)", 30, 80, 50, 1,
                              help="Characteristic life in months")

    st.markdown("---")
    st.markdown("### RUL Distribution")
    gamma_shape = st.slider("Gamma Shape (α)", 3.0, 15.0, 8.0, 0.5,
                            help="Shape of the posterior RUL distribution")
    gamma_scale = st.slider("Gamma Scale (β)", 1.0, 4.0, 2.3, 0.1,
                            help="Scale of the posterior RUL distribution")

    st.markdown("---")
    st.caption("Proof of Concept · Synthetic Data · Feb 2026")


# ── Helper: Plotly Layout ────────────────────────────────────────────
def base_layout(title="", xaxis_title="", yaxis_title="", height=480):
    return dict(
        title=dict(text=title, font=dict(size=18, color=NAVY, family="DM Sans")),
        font=dict(family="DM Sans", size=12, color=DARK_GRAY),
        plot_bgcolor=BG,
        paper_bgcolor="white",
        height=height,
        margin=dict(l=60, r=30, t=60, b=60),
        xaxis=dict(title=xaxis_title, gridcolor="#e8e8e8", zerolinecolor="#ddd"),
        yaxis=dict(title=yaxis_title, gridcolor="#e8e8e8", zerolinecolor="#ddd"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ddd", borderwidth=1),
    )


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("# 🔧 Remaining Useful Life — Coolant Pump Fleet")
st.markdown(
    "Interactive proof-of-concept dashboard for the Bayesian hierarchical RUL prediction model. "
    "Adjust parameters in the sidebar to explore how model inputs affect predictions."
)

# ══════════════════════════════════════════════════════════════════════
# ROW 1: Fleet Status Cards
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Fleet Health Overview")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="metric-card healthy">
        <h3>Healthy</h3>
        <div class="value">23</div>
        <div class="sub">12+ months RUL</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card watch">
        <h3>Watch List</h3>
        <div class="value">6</div>
        <div class="sub">3–12 months RUL</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card critical">
        <h3>Critical</h3>
        <div class="value">2</div>
        <div class="sub">&lt; 3 months RUL</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="metric-card offline">
        <h3>Offline</h3>
        <div class="value">1</div>
        <div class="sub">No data</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ROW 2: Component Table
# ══════════════════════════════════════════════════════════════════════
import pandas as pd

components_df = pd.DataFrame([
    {"Component": "CP-001", "Location": "Reactor Bldg A", "Installed": "2021-03-15", "Age (mo)": 59, "Median RUL": 18.2, "90% CI": "(12.4 – 27.1)", "Status": "🟢 Healthy"},
    {"Component": "CP-002", "Location": "Reactor Bldg A", "Installed": "2020-11-08", "Age (mo)": 63, "Median RUL": 24.6, "90% CI": "(18.2 – 34.5)", "Status": "🟢 Healthy"},
    {"Component": "CP-003", "Location": "Turbine Hall",   "Installed": "2019-06-22", "Age (mo)": 79, "Median RUL": 5.8,  "90% CI": "(2.1 – 10.3)",  "Status": "🟡 Watch"},
    {"Component": "CP-004", "Location": "Reactor Bldg B", "Installed": "2021-09-01", "Age (mo)": 45, "Median RUL": 31.4, "90% CI": "(22.7 – 42.8)", "Status": "🟢 Healthy"},
    {"Component": "CP-005", "Location": "Aux Building",   "Installed": "2020-01-17", "Age (mo)": 72, "Median RUL": 2.1,  "90% CI": "(0.4 – 4.8)",   "Status": "🔴 Critical"},
    {"Component": "CP-006", "Location": "Turbine Hall",   "Installed": "2022-04-30", "Age (mo)": 38, "Median RUL": 28.9, "90% CI": "(20.1 – 38.6)", "Status": "🟢 Healthy"},
    {"Component": "CP-007", "Location": "Reactor Bldg B", "Installed": "2019-12-10", "Age (mo)": 68, "Median RUL": 8.4,  "90% CI": "(4.2 – 14.1)",  "Status": "🟡 Watch"},
    {"Component": "CP-008", "Location": "Aux Building",   "Installed": "2021-07-20", "Age (mo)": 52, "Median RUL": 1.8,  "90% CI": "(0.2 – 4.1)",   "Status": "🔴 Critical"},
])

st.dataframe(
    components_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Median RUL": st.column_config.NumberColumn(format="%.1f mo"),
        "Age (mo)": st.column_config.NumberColumn(format="%d"),
    }
)


# ══════════════════════════════════════════════════════════════════════
# ROW 3: RUL Distribution + Degradation Trajectory
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
col_left, col_right = st.columns(2)

# ── RUL Probability Distribution ─────────────────────────────────────
with col_left:
    st.markdown("### RUL Probability Distribution")

    np.random.seed(42)
    rul_samples = gamma.rvs(a=gamma_shape, scale=gamma_scale, size=10000)
    x_range = np.linspace(0, max(rul_samples.max(), 50), 500)
    pdf_vals = gamma.pdf(x_range, a=gamma_shape, scale=gamma_scale)

    p5 = np.percentile(rul_samples, 5)
    p95 = np.percentile(rul_samples, 95)
    median_rul = np.median(rul_samples)

    fig_rul = go.Figure()

    # Critical zone
    fig_rul.add_vrect(x0=0, x1=3, fillcolor=RED, opacity=0.07,
                      line_width=0, annotation_text="Critical<br>Zone",
                      annotation_position="top left",
                      annotation=dict(font=dict(size=10, color=RED)))

    # 90% CI shading
    mask = (x_range >= p5) & (x_range <= p95)
    fig_rul.add_trace(go.Scatter(
        x=np.concatenate([x_range[mask], x_range[mask][::-1]]),
        y=np.concatenate([pdf_vals[mask], np.zeros(mask.sum())]),
        fill='toself', fillcolor=f'rgba(46,117,182,0.2)',
        line=dict(width=0),
        name=f'90% PI ({p5:.1f} – {p95:.1f} mo)',
        hoverinfo='skip',
    ))

    # Full distribution
    fig_rul.add_trace(go.Scatter(
        x=x_range, y=pdf_vals,
        mode='lines', line=dict(color=ACCENT, width=2.5),
        name='Posterior Distribution',
    ))

    # Median line
    fig_rul.add_vline(x=median_rul, line_dash="dash", line_color=NAVY, line_width=2,
                      annotation_text=f"Median: {median_rul:.1f} mo",
                      annotation_position="top")

    fig_rul.update_layout(**base_layout(
        title="Component CP-005 — RUL Posterior",
        xaxis_title="Remaining Useful Life (months)",
        yaxis_title="Probability Density",
    ))
    st.plotly_chart(fig_rul, use_container_width=True)


# ── Degradation Trajectory ───────────────────────────────────────────
with col_right:
    st.markdown("### Degradation Trajectory & Projection")

    np.random.seed(7)
    months_hist = np.arange(0, 59)
    base_deg = 100 - 0.3 * months_hist - 0.004 * months_hist**2
    noise_hist = np.cumsum(np.random.normal(0, 0.3, len(months_hist)))
    health_hist = np.clip(base_deg + noise_hist, 0, 100)

    months_future = np.arange(58, 90)
    n_traj = 300
    trajectories = []
    for _ in range(n_traj):
        rate = np.random.normal(deg_rate, 0.12)
        accel = np.random.normal(deg_accel, 0.002)
        nf = np.cumsum(np.random.normal(0, noise_std, len(months_future)))
        traj = health_hist[-1] - rate * (months_future - 58) - accel * (months_future - 58)**2 + nf
        trajectories.append(traj)
    trajectories = np.array(trajectories)

    med_traj = np.median(trajectories, axis=0)
    p10 = np.percentile(trajectories, 10, axis=0)
    p25 = np.percentile(trajectories, 25, axis=0)
    p75 = np.percentile(trajectories, 75, axis=0)
    p90 = np.percentile(trajectories, 90, axis=0)

    fig_deg = go.Figure()

    # Prediction envelopes
    fig_deg.add_trace(go.Scatter(
        x=np.concatenate([months_future, months_future[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself', fillcolor='rgba(46,117,182,0.1)',
        line=dict(width=0), name='80% Envelope', hoverinfo='skip',
    ))
    fig_deg.add_trace(go.Scatter(
        x=np.concatenate([months_future, months_future[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself', fillcolor='rgba(46,117,182,0.2)',
        line=dict(width=0), name='50% Envelope', hoverinfo='skip',
    ))

    # Historical
    fig_deg.add_trace(go.Scatter(
        x=months_hist, y=health_hist,
        mode='lines', line=dict(color=ACCENT, width=2.5),
        name='Observed Health Index',
    ))

    # Predicted median
    fig_deg.add_trace(go.Scatter(
        x=months_future, y=med_traj,
        mode='lines', line=dict(color=NAVY, width=2, dash='dash'),
        name='Predicted Median',
    ))

    # Failure threshold
    fig_deg.add_hline(y=failure_threshold, line_dash="dot", line_color=RED, line_width=2,
                      annotation_text=f"Failure Threshold (HI={failure_threshold})",
                      annotation_position="top left",
                      annotation=dict(font=dict(color=RED, size=10)))

    # NOW line
    fig_deg.add_vline(x=58, line_color=MED_GRAY, line_width=1.5,
                      annotation_text="NOW", annotation_position="top")

    fig_deg.update_layout(**base_layout(
        title="Component CP-003 — Degradation Projection",
        xaxis_title="Component Age (months)",
        yaxis_title="Health Index",
    ))
    fig_deg.update_yaxis(range=[15, 105])
    st.plotly_chart(fig_deg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# ROW 4: Survival Curve + Feature Importance
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
col_left2, col_right2 = st.columns(2)

# ── Population Survival Analysis ─────────────────────────────────────
with col_left2:
    st.markdown("### Population Survival Analysis")

    t = np.linspace(0, 80, 500)
    surv_model = weibull_min.sf(t, c=weibull_shape, scale=weibull_scale)
    surv_upper = weibull_min.sf(t, c=weibull_shape - 0.5, scale=weibull_scale + 3)
    surv_lower = weibull_min.sf(t, c=weibull_shape + 0.5, scale=weibull_scale - 3)

    # KM steps (simulated)
    np.random.seed(123)
    failure_times = np.sort(weibull_min.rvs(c=weibull_shape, scale=weibull_scale, size=45))
    n_total = 75
    km_times = [0]
    km_surv = [1.0]
    for ft in failure_times:
        at_risk = n_total - np.sum(failure_times < ft)
        if at_risk <= 0:
            break
        km_times.append(ft)
        km_surv.append(km_surv[-1] * (1 - 1 / max(at_risk, 1)))
    km_times.append(80)
    km_surv.append(km_surv[-1])

    fig_surv = go.Figure()

    # Confidence band
    fig_surv.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([surv_upper, surv_lower[::-1]]),
        fill='toself', fillcolor='rgba(46,117,182,0.12)',
        line=dict(width=0), name='95% Confidence Band', hoverinfo='skip',
    ))

    # KM estimate
    fig_surv.add_trace(go.Scatter(
        x=km_times, y=km_surv,
        mode='lines', line=dict(color=DARK_GRAY, width=1.5, shape='hv'),
        name='Kaplan-Meier (observed)',
    ))

    # Model prediction
    fig_surv.add_trace(go.Scatter(
        x=t, y=surv_model,
        mode='lines', line=dict(color=ACCENT, width=2.5),
        name='Bayesian Model Prediction',
    ))

    # Key percentile markers
    for pct in [0.75, 0.50, 0.25]:
        idx = np.argmin(np.abs(surv_model - pct))
        t_pct = t[idx]
        fig_surv.add_trace(go.Scatter(
            x=[t_pct], y=[pct], mode='markers+text',
            marker=dict(size=8, color=NAVY),
            text=[f'{pct:.0%}: {t_pct:.0f} mo'], textposition='top right',
            textfont=dict(size=10, color=DARK_GRAY),
            showlegend=False,
        ))

    fig_surv.update_layout(**base_layout(
        title=f"Coolant Pump Fleet Survival (β={weibull_shape}, η={weibull_scale})",
        xaxis_title="Component Age (months)",
        yaxis_title="Survival Probability",
    ))
    fig_surv.update_yaxis(range=[0, 1.05])
    st.plotly_chart(fig_surv, use_container_width=True)


# ── Degradation Drivers ──────────────────────────────────────────────
with col_right2:
    st.markdown("### Degradation Drivers (SHAP)")

    features = [
        "Load Factor (avg)", "Ambient Temperature", "Operating Hours",
        "Seal Leak Rate", "Coolant Flow Rate", "Motor Current Draw",
        "Discharge Pressure (Δ)", "Vibration Kurtosis",
        "Vibration RMS (X)", "Bearing Temp (trend)",
    ]
    importance = np.array([0.026, 0.035, 0.048, 0.065, 0.072, 0.089, 0.118, 0.142, 0.187, 0.218])
    colors_bar = [RED if v > 0.15 else AMBER if v > 0.08 else ACCENT for v in importance]

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        y=features, x=importance,
        orientation='h',
        marker_color=colors_bar,
        text=[f'{v:.1%}' for v in importance],
        textposition='outside',
        textfont=dict(size=11, color=DARK_GRAY),
    ))
    fig_shap.update_layout(**base_layout(
        title="Component CP-005 — Feature Importance",
        xaxis_title="Relative Importance (SHAP-based)",
        height=480,
    ))
    fig_shap.update_layout(showlegend=False)
    fig_shap.update_xaxis(range=[0, 0.28])
    st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# ROW 5: Maintenance Timeline + Model Validation
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
col_left3, col_right3 = st.columns(2)

# ── Maintenance Planning Timeline ────────────────────────────────────
with col_left3:
    st.markdown("### Maintenance Planning Timeline")

    gantt_data = [
        ("CP-008", 1.8, 0.2, 4.1, RED),
        ("CP-005", 2.1, 0.4, 4.8, RED),
        ("CP-003", 5.8, 2.1, 10.3, AMBER),
        ("CP-007", 8.4, 4.2, 14.1, AMBER),
        ("CP-009", 9.2, 5.5, 13.8, AMBER),
        ("CP-012", 10.1, 6.0, 15.5, AMBER),
        ("CP-015", 11.8, 7.1, 17.2, AMBER),
        ("CP-011", 14.5, 9.2, 20.8, AMBER),
        ("CP-001", 18.2, 12.4, 27.1, GREEN),
        ("CP-010", 20.5, 14.1, 29.3, GREEN),
        ("CP-002", 24.6, 18.2, 34.5, GREEN),
        ("CP-004", 31.4, 22.7, 42.8, GREEN),
        ("CP-006", 28.9, 20.1, 38.6, GREEN),
    ]

    fig_gantt = go.Figure()

    for i, (cid, med, lo, hi, color) in enumerate(gantt_data):
        # CI bar
        fig_gantt.add_trace(go.Bar(
            y=[cid], x=[hi - lo], base=[lo],
            orientation='h',
            marker_color=color, opacity=0.25,
            showlegend=False, hoverinfo='skip',
        ))
        # Median marker
        fig_gantt.add_trace(go.Scatter(
            x=[med], y=[cid],
            mode='markers',
            marker=dict(size=10, color=color, symbol='diamond',
                        line=dict(width=1.5, color='white')),
            showlegend=False,
            hovertemplate=f'{cid}: Median {med:.1f} mo<br>90% CI: ({lo:.1f} – {hi:.1f})<extra></extra>',
        ))

    # Outage windows
    fig_gantt.add_vrect(x0=3, x1=5, fillcolor=NAVY, opacity=0.06, line_width=0,
                        annotation_text="Spring<br>Outage", annotation_position="top left",
                        annotation=dict(font=dict(size=9, color=NAVY)))
    fig_gantt.add_vrect(x0=15, x1=17, fillcolor=NAVY, opacity=0.06, line_width=0,
                        annotation_text="Fall<br>Outage", annotation_position="top left",
                        annotation=dict(font=dict(size=9, color=NAVY)))

    fig_gantt.update_layout(**base_layout(
        title="Predicted Replacement Windows",
        xaxis_title="Months from Now",
        height=520,
    ))
    fig_gantt.update_layout(barmode='overlay')
    fig_gantt.update_yaxis(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=11))
    st.plotly_chart(fig_gantt, use_container_width=True)


# ── Model Validation ─────────────────────────────────────────────────
with col_right3:
    st.markdown("### Model Validation Results")

    fig_val = make_subplots(rows=1, cols=2,
                            subplot_titles=("Predicted vs. Actual RUL", "Model Comparison — MAE"),
                            horizontal_spacing=0.15)

    # Scatter: predicted vs actual
    np.random.seed(99)
    n_test = 30
    actual = np.random.uniform(2, 40, n_test)
    predicted = np.clip(actual + np.random.normal(0, 3.5, n_test), 0, 50)
    errors = np.abs(predicted - actual)
    scatter_colors = [GREEN if e < 3 else AMBER if e < 6 else RED for e in errors]

    fig_val.add_trace(go.Scatter(
        x=actual, y=predicted, mode='markers',
        marker=dict(size=8, color=scatter_colors, line=dict(width=0.8, color='white')),
        name='Predictions', showlegend=False,
    ), row=1, col=1)

    # Perfect prediction line
    fig_val.add_trace(go.Scatter(
        x=[0, 45], y=[0, 45], mode='lines',
        line=dict(color=MED_GRAY, dash='dash', width=1.5),
        name='Perfect', showlegend=False,
    ), row=1, col=1)

    # Model comparison bars
    models = ['Bayesian<br>Hierarchical', 'Hybrid<br>(Bayes+ML)', 'XGBoost',
              'Random<br>Surv Forest', 'Weibull<br>Baseline']
    mae_vals = [3.2, 2.8, 3.9, 4.1, 6.5]
    bar_colors = [ACCENT, GREEN, ACCENT, ACCENT, ACCENT]

    fig_val.add_trace(go.Bar(
        x=models, y=mae_vals,
        marker_color=bar_colors, opacity=0.8,
        text=[f'{v:.1f}' for v in mae_vals], textposition='outside',
        textfont=dict(size=11, color=DARK_GRAY),
        name='MAE (months)', showlegend=False,
    ), row=1, col=2)

    fig_val.update_layout(
        font=dict(family="DM Sans", size=12, color=DARK_GRAY),
        plot_bgcolor=BG, paper_bgcolor="white",
        height=520, margin=dict(l=50, r=30, t=60, b=60),
    )
    fig_val.update_xaxes(title_text="Actual RUL (months)", row=1, col=1, gridcolor="#e8e8e8")
    fig_val.update_yaxes(title_text="Predicted RUL (months)", row=1, col=1, gridcolor="#e8e8e8")
    fig_val.update_xaxes(row=1, col=2, gridcolor="#e8e8e8")
    fig_val.update_yaxes(title_text="MAE (months)", row=1, col=2, gridcolor="#e8e8e8", range=[0, 8])

    st.plotly_chart(fig_val, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# ROW 6: Methodology Summary
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Methodology Overview")

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("""
    **Data Inputs**
    - Daily sensor readings (vibration, temperature, pressure, flow)
    - Maintenance event history (failures, repairs, inspections)
    - Operating context (load, duty cycle, plant mode)
    - Component metadata (manufacturer, install date, material)
    """)
with m2:
    st.markdown("""
    **Modelling Approach**
    - Hierarchical Bayesian survival model with component-level random effects
    - XGBoost feature engineering for health index construction
    - SHAP-based explainability for degradation driver identification
    - Hybrid ensemble achieving MAE of 2.8 months with 91% CI coverage
    """)
with m3:
    st.markdown("""
    **Dashboard Outputs**
    - Fleet-wide health overview with status classification
    - Per-component RUL posterior distributions with uncertainty
    - Degradation trajectory projections with confidence fans
    - Maintenance planning timelines aligned to outage windows
    """)

st.markdown("---")
st.caption("RUL Prediction Dashboard · Proof of Concept · All data is synthetic · Built with Streamlit & Plotly")
