"""
RUL Prediction Dashboard — Proof of Concept
Remaining Useful Life prediction for industrial coolant pump assemblies.
Bayesian hierarchical model with interactive parameter exploration.

Every view is driven by one simulated fleet: each pump has a fixed observed
health history, and the sidebar parameters control how the model projects
that history forward. RUL posteriors are the first-passage times of those
projected paths through the failure threshold.
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy.stats import gaussian_kde

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Prediction — Coolant Pump Fleet",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme Colors ─────────────────────────────────────────────────────
INK = "#0F172A"
SLATE = "#475569"
MUTED = "#94A3B8"
GRID = "#E8EDF4"
NAVY = "#1E3A5F"
BLUE = "#2563EB"
LIGHT_BLUE = "#93C5FD"
GREEN = "#16A34A"
AMBER = "#D97706"
RED = "#DC2626"
GRAY = "#64748B"
FONT = "Inter, 'Segoe UI', sans-serif"

STATUS_COLORS = {"Critical": RED, "Watch": AMBER, "Healthy": GREEN, "Offline": GRAY}


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── Model Constants ──────────────────────────────────────────────────
AS_OF = pd.Timestamp("2026-02-01")   # data snapshot date
FLEET_SIZE = 32
OFFLINE_IDS = {"CP-019"}             # sensor feed lost
CRITICAL_MO = 3
WATCH_MO = 12
HORIZON = 120                        # projection horizon (months)
N_TRAJ = 500
DAYS_PER_MONTH = 30.4375
LOCATIONS = ["Reactor Bldg A", "Reactor Bldg B", "Turbine Hall", "Aux Building", "Intake Structure"]

# Planned outage windows, in months from the snapshot date
OUTAGES = [
    ("Spring 2026", 2.0, 3.5),
    ("Fall 2026", 8.0, 9.5),
    ("Spring 2027", 14.0, 15.5),
    ("Fall 2027", 20.0, 21.5),
    ("Spring 2028", 26.0, 27.5),
]

# Parameters the observed histories were generated with
DEFAULTS = dict(
    deg_rate=0.40,
    deg_accel=0.005,
    noise_std=0.35,
    threshold=40,
    wb_shape=3.5,
    wb_scale=50,
)


def months_to_date(m):
    return AS_OF + pd.to_timedelta(np.asarray(m) * DAYS_PER_MONTH, unit="D")


def fmt_rul(x):
    return f">{HORIZON}" if x >= HORIZON else f"{x:.1f}"


# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .block-container {{
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* White cards on the tinted page background */
    div[class*="st-key-card"],
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}

    .hero {{
        background: linear-gradient(120deg, #0F2440 0%, #1E3A5F 55%, #2563EB 130%);
        border-radius: 16px;
        padding: 1.8rem 2rem 1.6rem;
        color: #F8FAFC;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(15, 36, 64, 0.18);
    }}
    .hero .eyebrow {{
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {LIGHT_BLUE};
        font-weight: 600;
    }}
    .hero h1 {{
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0 0.4rem;
        padding: 0;
        line-height: 1.2;
    }}
    .hero p {{
        color: #CBD5E1;
        margin: 0 0 0.9rem;
        max-width: 760px;
        font-size: 0.95rem;
    }}
    .chip {{
        display: inline-block;
        font-size: 0.75rem;
        padding: 0.2rem 0.65rem;
        margin: 0 0.4rem 0.3rem 0;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        color: #E2E8F0;
    }}

    .kpi {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.15rem 0.95rem;
        height: 100%;
    }}
    .kpi .label {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: {SLATE};
    }}
    .kpi .dot {{
        width: 9px; height: 9px; border-radius: 50%;
        display: inline-block;
    }}
    .kpi .value {{
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.15;
        margin-top: 0.35rem;
        color: {INK};
        font-variant-numeric: tabular-nums;
    }}
    .kpi .value small {{
        font-size: 0.9rem;
        font-weight: 500;
        color: {MUTED};
        margin-left: 0.3rem;
    }}
    .kpi .sub {{
        font-size: 0.8rem;
        color: {SLATE};
        margin-top: 0.15rem;
    }}
    .kpi .bar {{
        height: 4px;
        border-radius: 4px;
        background: {GRID};
        margin-top: 0.7rem;
        overflow: hidden;
    }}
    .kpi .bar span {{ display: block; height: 100%; border-radius: 4px; }}

    .card-head .title {{
        font-size: 1.02rem;
        font-weight: 650;
        color: {INK};
    }}
    .card-head .sub {{
        font-size: 0.82rem;
        color: {SLATE};
        margin-top: 0.1rem;
    }}

    .section-label {{
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {MUTED};
        margin: 1.2rem 0 0.5rem;
    }}

    button[role="tab"] p {{ font-size: 0.95rem; font-weight: 600; }}
    div[data-testid="stMetric"] {{ padding: 0.9rem 1rem; }}
    div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
</style>
""", unsafe_allow_html=True)


# ── Plotly Template ──────────────────────────────────────────────────
_axis = dict(
    gridcolor=GRID, zeroline=False, linecolor="#CBD5E1", showline=True,
    ticks="outside", tickcolor="#CBD5E1", ticklen=4, automargin=True,
    title=dict(font=dict(size=12, color=SLATE), standoff=10),
)
pio.templates["rul"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, size=12, color=SLATE),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=[BLUE, NAVY, AMBER, GREEN, RED],
    margin=dict(l=8, r=16, t=36, b=8),
    xaxis=_axis,
    yaxis=_axis,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
))
pio.templates.default = "rul"
PLOT_CONFIG = {"displayModeBar": False}


def chart_card(title, subtitle, fig, height=380):
    with st.container(border=True, key="card_" + re.sub(r"\W+", "_", title)):
        st.markdown(
            f'<div class="card-head"><div class="title">{title}</div>'
            f'<div class="sub">{subtitle}</div></div>',
            unsafe_allow_html=True,
        )
        fig.update_layout(height=height, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG)


def kpi_tile(label, value, sub, color, share=None, unit=""):
    bar = ""
    if share is not None:
        bar = f'<div class="bar"><span style="width:{share * 100:.0f}%;background:{color}"></span></div>'
    unit_html = f"<small>{unit}</small>" if unit else ""
    return (
        f'<div class="kpi"><div class="label"><span class="dot" style="background:{color}"></span>{label}</div>'
        f'<div class="value">{value}{unit_html}</div><div class="sub">{sub}</div>{bar}</div>'
    )


# ── Sidebar ──────────────────────────────────────────────────────────
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def reset_params():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("## ⚙️ Model Parameters")
    st.caption("Adjust the model assumptions. Observed histories stay fixed; "
               "projections, RUL posteriors and fleet status update live.")

    st.markdown("#### Degradation model")
    deg_rate = st.slider("Degradation rate (HI / month)", 0.10, 1.00, step=0.05, key="deg_rate",
                         help="Population-average linear decline of the health index per month")
    deg_accel = st.slider("Degradation acceleration", 0.001, 0.015, step=0.001, format="%.3f",
                          key="deg_accel", help="Quadratic wear-out term: degradation speeds up with age")
    noise_std = st.slider("Process noise (σ)", 0.10, 1.00, step=0.05, key="noise_std",
                          help="Month-to-month stochastic variability of the degradation path")

    st.markdown("#### Failure definition")
    failure_threshold = st.slider("Health index threshold", 20, 60, step=5, key="threshold",
                                  help="A pump is considered failed once its health index drops below this")

    st.markdown("#### Population prior (Weibull)")
    weibull_shape = st.slider("Shape β", 1.5, 6.0, step=0.1, key="wb_shape",
                              help="Higher β = failures concentrated around the characteristic life")
    weibull_scale = st.slider("Scale η (months)", 30, 80, step=1, key="wb_scale",
                              help="Characteristic life: age by which 63.2% of pumps have failed")

    st.button("↺  Reset to defaults", on_click=reset_params, width="stretch")
    st.markdown("---")
    st.caption(f"Proof of concept · Synthetic data · Snapshot {AS_OF:%b %Y}")


# ══════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_fleet():
    """Static fleet metadata, component random effects and observed histories."""
    rng = np.random.default_rng(120)
    rows, histories = [], {}
    d = DEFAULTS
    for i in range(FLEET_SIZE):
        cid = f"CP-{i + 1:03d}"
        age = int(rng.integers(30, 74))
        mult = float(rng.lognormal(0.0, 0.18))   # component-level random effect
        installed = AS_OF - pd.DateOffset(months=age) - pd.Timedelta(days=int(rng.integers(0, 28)))

        t = np.arange(age + 1)
        walk = np.concatenate([[0.0], np.cumsum(rng.normal(0, 1, age))]) * d["noise_std"] * 0.6
        hi = np.clip(100 - mult * (d["deg_rate"] * t + d["deg_accel"] * t**2) + walk, 0, 100)

        # Sensor-derived observations scatter around the latent health state
        meas = np.random.default_rng(500 + i).normal(0, 0.9, age + 1)
        histories[cid] = np.clip(hi + meas, 0, 100)
        rows.append(dict(
            id=cid,
            location=LOCATIONS[int(rng.integers(len(LOCATIONS)))],
            installed=installed.date(),
            age=age,
            mult=mult,
            hi=float(hi[-1]),
            offline=cid in OFFLINE_IDS,
        ))
    return pd.DataFrame(rows), histories


def first_crossing(paths, threshold):
    """Fractional month at which each path first drops below threshold (HORIZON if never)."""
    below = paths < threshold
    hit = below.any(axis=1)
    idx = np.argmax(below, axis=1)
    rows = np.arange(len(paths))
    prev = paths[rows, np.maximum(idx - 1, 0)]
    cur = paths[rows, idx]
    frac = np.where(idx > 0, (prev - threshold) / np.maximum(prev - cur, 1e-9), 0.0)
    rul = np.where(idx > 0, idx - 1 + frac, 0.0)
    return np.where(hit, rul, float(HORIZON))


def target_outage(p10):
    """Latest planned outage the pump survives to with ≥90% probability."""
    if p10 > 24:
        return "Routine monitoring"
    feasible = [name for name, _, end in OUTAGES if end <= p10]
    return feasible[-1] if feasible else "Expedite"


@st.cache_data(show_spinner=False)
def simulate_fleet(rate, accel, noise, threshold):
    fleet, _ = build_fleet()
    k = np.arange(HORIZON + 1)
    sims, summary = {}, []
    for i, row in enumerate(fleet.itertuples()):
        if row.offline:
            summary.append(dict(id=row.id, status="Offline"))
            continue
        rng = np.random.default_rng(1000 + i)
        m = row.mult * np.exp(rng.normal(0, 0.10, (N_TRAJ, 1)))     # posterior on random effect
        a = accel * np.exp(rng.normal(0, 0.20, (N_TRAJ, 1)))         # uncertainty on wear-out term
        steps = rng.normal(0, 1, (N_TRAJ, HORIZON)) * noise
        walk = np.hstack([np.zeros((N_TRAJ, 1)), np.cumsum(steps, axis=1)])
        drop = m * (rate * k + a * ((row.age + k) ** 2 - row.age ** 2))
        paths = row.hi - drop + walk

        rul = first_crossing(paths, threshold)
        p05, p10, p50, p95 = np.percentile(rul, [5, 10, 50, 95])
        status = "Critical" if p50 < CRITICAL_MO else "Watch" if p50 < WATCH_MO else "Healthy"
        sims[row.id] = dict(
            rul=rul,
            bands=np.percentile(paths, [10, 25, 50, 75, 90], axis=0),
        )
        summary.append(dict(
            id=row.id, status=status, median=p50, p05=p05, p10=p10, p95=p95,
            p_fail_6=float(np.mean(rul <= 6)), p_fail_12=float(np.mean(rul <= 12)),
            target=target_outage(p10),
        ))
    summary = fleet.merge(pd.DataFrame(summary), on="id")
    return summary, sims


fleet_meta, histories = build_fleet()
fleet, sims = simulate_fleet(deg_rate, deg_accel, noise_std, failure_threshold)
active = fleet[~fleet["offline"]].sort_values("median")
counts = fleet["status"].value_counts()
expected_12 = active["p_fail_12"].sum()


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">Predictive maintenance · Proof of concept</div>
    <h1>Remaining Useful Life — Coolant Pump Fleet</h1>
    <p>Bayesian hierarchical degradation model for {FLEET_SIZE} reactor coolant pump assemblies.
    Tune the model in the sidebar and every forecast, status and maintenance window below re-computes.</p>
    <span class="chip">📅 Snapshot {AS_OF:%d %b %Y}</span>
    <span class="chip">🎲 {N_TRAJ} Monte Carlo paths / pump</span>
    <span class="chip">🧪 Synthetic data</span>
</div>
""", unsafe_allow_html=True)

# ── Fleet KPIs ───────────────────────────────────────────────────────
tiles = [
    ("Healthy", f"{WATCH_MO}+ months median RUL"),
    ("Watch", f"{CRITICAL_MO}–{WATCH_MO} months median RUL"),
    ("Critical", f"&lt; {CRITICAL_MO} months median RUL"),
    ("Offline", "No sensor data"),
]
cols = st.columns(5)
for col, (status, sub) in zip(cols, tiles):
    n = int(counts.get(status, 0))
    col.markdown(kpi_tile(status, n, sub, STATUS_COLORS[status], share=n / FLEET_SIZE,
                          unit=f"/ {FLEET_SIZE}"), unsafe_allow_html=True)
cols[4].markdown(kpi_tile("Expected failures", f"{expected_12:.1f}",
                          "Next 12 months, if no action is taken", NAVY), unsafe_allow_html=True)

st.write("")
tab_fleet, tab_comp, tab_pop, tab_method = st.tabs(
    ["Fleet overview", "Component deep-dive", "Population & validation", "Methodology"]
)


# ══════════════════════════════════════════════════════════════════════
# TAB 1: Fleet Overview
# ══════════════════════════════════════════════════════════════════════
with tab_fleet:
    # ── Maintenance Planning Timeline ────────────────────────────────
    plan = active.head(15)
    fig_gantt = go.Figure()

    for name, start, end in OUTAGES:
        fig_gantt.add_vrect(
            x0=start, x1=end, fillcolor=NAVY, opacity=0.08, line_width=0, layer="below",
            annotation_text=name, annotation_position="top left",
            annotation=dict(font=dict(size=10, color=NAVY)),
        )

    for status in ["Critical", "Watch", "Healthy"]:
        sub = plan[plan["status"] == status]
        if sub.empty:
            continue
        color = STATUS_COLORS[status]
        hi = sub["p95"].clip(upper=36)
        fig_gantt.add_trace(go.Bar(
            y=sub["id"], base=sub["p05"], x=hi - sub["p05"],
            orientation="h", marker=dict(color=rgba(color, 0.22), line=dict(width=0)),
            width=0.55, showlegend=False, hoverinfo="skip",
        ))
        fig_gantt.add_trace(go.Scatter(
            x=sub["median"], y=sub["id"], mode="markers", name=status,
            marker=dict(size=11, color=color, symbol="diamond", line=dict(width=1.5, color="white")),
            customdata=sub[["p05", "p95", "target"]].to_numpy(dtype=object),
            hovertemplate="<b>%{y}</b><br>Median RUL %{x:.1f} mo"
                          "<br>90% PI %{customdata[0]:.1f} – %{customdata[1]:.1f} mo"
                          "<br>Plan: %{customdata[2]}<extra></extra>",
        ))

    ticks = np.arange(0, 31, 3)
    fig_gantt.add_vline(x=0, line_color=SLATE, line_width=1.5)
    fig_gantt.update_layout(barmode="overlay", bargap=0.35)
    fig_gantt.update_xaxes(range=[-0.7, 30], tickvals=ticks,
                           ticktext=[f"{d:%b %Y}" for d in months_to_date(ticks)])
    fig_gantt.update_yaxes(autorange="reversed", showgrid=False,
                           tickfont=dict(family="JetBrains Mono, monospace", size=11, color=INK))

    chart_card(
        "Predicted failure windows vs. planned outages",
        "15 most at-risk pumps · ◆ median failure date · bar = 90% prediction interval · "
        "shaded = planned outage windows",
        fig_gantt, height=520,
    )

    # ── Component Table ──────────────────────────────────────────────
    st.markdown('<div class="section-label">Component register</div>', unsafe_allow_html=True)
    status_order = {"Critical": 0, "Watch": 1, "Healthy": 2, "Offline": 3}
    table = (
        fleet.assign(_o=fleet["status"].map(status_order))
        .sort_values(["_o", "median"])
        .assign(
            interval=lambda d: [
                "—" if pd.isna(lo) else f"{fmt_rul(lo)} – {fmt_rul(hi)}"
                for lo, hi in zip(d["p05"], d["p95"])
            ],
            p_fail_6=lambda d: d["p_fail_6"] * 100,
            hi=lambda d: np.where(d["offline"], np.nan, d["hi"]),
            target=lambda d: d["target"].fillna("Restore sensor feed"),
        )
        [["id", "status", "location", "age", "hi", "median", "interval", "p_fail_6", "target", "installed"]]
    )
    styled = table.style.map(
        lambda s: f"color: {STATUS_COLORS[s]}; font-weight: 600", subset=["status"]
    )
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=420,
        column_config={
            "id": st.column_config.TextColumn("Component"),
            "location": st.column_config.TextColumn("Location"),
            "installed": st.column_config.DateColumn("Installed", format="YYYY-MM-DD"),
            "age": st.column_config.NumberColumn("Age (mo)", format="%d"),
            "hi": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%.0f",
                                                  width="small"),
            "median": st.column_config.NumberColumn("Median RUL (mo)", format="%.1f"),
            "interval": st.column_config.TextColumn("90% PI (mo)"),
            "p_fail_6": st.column_config.ProgressColumn("P(fail ≤ 6 mo)", min_value=0, max_value=100,
                                                        format="%.0f%%"),
            "target": st.column_config.TextColumn("Plan replacement at"),
            "status": st.column_config.TextColumn("Status"),
        },
    )
    st.caption("Replacement target = latest planned outage the pump reaches with ≥ 90% probability.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2: Component Deep-Dive
# ══════════════════════════════════════════════════════════════════════
with tab_comp:
    ids = active["id"].tolist()
    labels = {r.id: f"{r.id} · {r.location} · {r.status}" for r in active.itertuples()}
    sel = st.selectbox("Component", ids, format_func=labels.get,
                       help="Sorted from most to least at-risk")
    row = active.set_index("id").loc[sel]
    sim = sims[sel]
    rul = sim["rul"]

    status_color = STATUS_COLORS[row["status"]]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_tile("Health index", f"{row['hi']:.0f}",
                         f"{row['hi'] - failure_threshold:+.0f} points vs. failure threshold",
                         status_color, share=row["hi"] / 100, unit="/ 100"), unsafe_allow_html=True)
    k2.markdown(kpi_tile("Median RUL", fmt_rul(row["median"]),
                         f"Expected failure ≈ {months_to_date(min(row['median'], HORIZON)):%b %Y}",
                         status_color, unit="months"), unsafe_allow_html=True)
    k3.markdown(kpi_tile("90% interval", f"{fmt_rul(row['p05'])}–{fmt_rul(row['p95'])}",
                         "5th – 95th percentile of RUL", BLUE, unit="months"), unsafe_allow_html=True)
    k4.markdown(kpi_tile("P(fail ≤ 6 months)", f"{row['p_fail_6']:.0%}", f"Plan: {row['target']}",
                         status_color, share=row["p_fail_6"]), unsafe_allow_html=True)
    st.write("")

    c_left, c_right = st.columns(2)

    # ── RUL Posterior ────────────────────────────────────────────────
    with c_left:
        fig_rul = go.Figure()
        x_max = float(min(max(np.percentile(rul, 99.5) * 1.3, 6), HORIZON))
        fig_rul.add_vrect(x0=0, x1=CRITICAL_MO, fillcolor=RED, opacity=0.07, line_width=0, layer="below")
        fig_rul.add_vrect(x0=CRITICAL_MO, x1=min(WATCH_MO, x_max), fillcolor=AMBER, opacity=0.06,
                          line_width=0, layer="below")

        if np.ptp(rul) < 1e-6:
            fig_rul.add_vline(x=float(rul[0]), line_color=BLUE, line_width=3)
        else:
            x = np.linspace(0, x_max, 400)
            pdf = gaussian_kde(rul)(x)
            # Reflect at zero so the density doesn't leak below 0 months
            pdf += gaussian_kde(-rul)(x) if rul.min() < 2 else 0
            inside = (x >= row["p05"]) & (x <= row["p95"])
            fig_rul.add_trace(go.Scatter(
                x=np.concatenate([x[inside], x[inside][::-1]]),
                y=np.concatenate([pdf[inside], np.zeros(inside.sum())]),
                fill="toself", fillcolor=rgba(BLUE, 0.16), line=dict(width=0),
                name="90% prediction interval", hoverinfo="skip",
            ))
            fig_rul.add_trace(go.Scatter(
                x=x, y=pdf, mode="lines", line=dict(color=BLUE, width=2.5), name="Posterior density",
                hovertemplate="%{x:.1f} mo<extra></extra>",
            ))
        fig_rul.add_vline(x=row["median"], line_dash="dash", line_color=NAVY, line_width=1.8,
                          annotation_text=f"Median {fmt_rul(row['median'])} mo",
                          annotation_position="top right",
                          annotation=dict(font=dict(color=NAVY, size=11)))
        fig_rul.update_xaxes(title_text="Remaining useful life (months)", range=[0, x_max])
        fig_rul.update_yaxes(title_text="Probability density", showticklabels=False)
        censored = np.mean(rul >= HORIZON)
        note = f" · {censored:.0%} of paths survive past {HORIZON} mo" if censored > 0.01 else ""
        chart_card(f"{sel} — RUL posterior",
                   f"First-passage time through HI = {failure_threshold} across {N_TRAJ} paths{note}",
                   fig_rul)

    # ── Degradation Trajectory ───────────────────────────────────────
    with c_right:
        hist = histories[sel]
        age = int(row["age"])
        p10, p25, p50, p75, p90 = sim["bands"]
        n_fwd = int(np.clip(np.ceil(row["p95"]) + 6, 18, HORIZON))
        fut = age + np.arange(n_fwd + 1)
        sl = slice(0, n_fwd + 1)

        fig_deg = go.Figure()
        fig_deg.add_trace(go.Scatter(
            x=np.concatenate([fut, fut[::-1]]), y=np.concatenate([p90[sl], p10[sl][::-1]]),
            fill="toself", fillcolor=rgba(BLUE, 0.10), line=dict(width=0),
            name="80% band", hoverinfo="skip",
        ))
        fig_deg.add_trace(go.Scatter(
            x=np.concatenate([fut, fut[::-1]]), y=np.concatenate([p75[sl], p25[sl][::-1]]),
            fill="toself", fillcolor=rgba(BLUE, 0.20), line=dict(width=0),
            name="50% band", hoverinfo="skip",
        ))
        fig_deg.add_trace(go.Scatter(
            x=np.arange(age + 1), y=hist, mode="lines", line=dict(color=NAVY, width=1.6),
            name="Observed", hovertemplate="Age %{x} mo · HI %{y:.1f}<extra></extra>",
        ))
        fig_deg.add_trace(go.Scatter(
            x=fut, y=p50[sl], mode="lines", line=dict(color=BLUE, width=2.2, dash="dash"),
            name="Projected median", hovertemplate="Age %{x} mo · HI %{y:.1f}<extra></extra>",
        ))
        if row["median"] < HORIZON:
            fig_deg.add_trace(go.Scatter(
                x=[age + row["median"]], y=[failure_threshold], mode="markers",
                marker=dict(size=11, color=RED, symbol="diamond", line=dict(width=1.5, color="white")),
                name="Median failure", hovertemplate="Median failure at age %{x:.1f} mo<extra></extra>",
            ))

        fig_deg.add_hline(y=failure_threshold, line_dash="dot", line_color=RED, line_width=1.6,
                          annotation_text=f"Failure threshold (HI {failure_threshold})",
                          annotation_position="bottom left",
                          annotation=dict(font=dict(color=RED, size=11)))
        fig_deg.add_vline(x=age, line_color=SLATE, line_width=1.2,
                          annotation_text="Now", annotation_position="top left",
                          annotation=dict(font=dict(color=SLATE, size=11)))
        y_low = max(0, min(failure_threshold, float(p10[n_fwd])) - 8)
        fig_deg.update_xaxes(title_text="Component age (months)")
        fig_deg.update_yaxes(title_text="Health index", range=[y_low, 103])
        chart_card(f"{sel} — degradation trajectory & projection",
                   f"{age} months of observed history · fan shows 50% and 80% projection bands",
                   fig_deg)

    # ── Degradation Drivers ──────────────────────────────────────────
    features = [
        "Bearing temperature (trend)", "Vibration RMS (X)", "Vibration kurtosis",
        "Discharge pressure (Δ)", "Motor current draw", "Coolant flow rate",
        "Seal leak rate", "Operating hours", "Ambient temperature", "Load factor (avg)",
    ]
    base_imp = np.array([0.218, 0.187, 0.142, 0.118, 0.089, 0.072, 0.065, 0.048, 0.035, 0.026])
    imp = base_imp * np.random.default_rng(int(sel[3:])).lognormal(0, 0.25, len(base_imp))
    imp = imp / imp.sum()
    order = np.argsort(imp)
    top3 = set(order[-3:])

    fig_shap = go.Figure(go.Bar(
        y=[features[i] for i in order], x=imp[order], orientation="h",
        marker=dict(color=[NAVY if i in top3 else LIGHT_BLUE for i in order]),
        text=[f"{v:.1%}" for v in imp[order]], textposition="outside",
        textfont=dict(size=11, color=INK), cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig_shap.update_xaxes(title_text="Share of mean |SHAP| contribution", tickformat=".0%",
                          range=[0, imp.max() * 1.18])
    fig_shap.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    fig_shap.update_layout(showlegend=False, margin=dict(t=12))
    chart_card(f"{sel} — what is driving degradation",
               "Relative feature contribution to the health-index model · top three drivers highlighted",
               fig_shap, height=400)


# ══════════════════════════════════════════════════════════════════════
# TAB 3: Population & Validation
# ══════════════════════════════════════════════════════════════════════
with tab_pop:
    p_left, p_right = st.columns(2)

    # ── Population Survival Analysis ─────────────────────────────────
    with p_left:
        t = np.linspace(0, 100, 400)

        def weibull_sf(tt, c, eta):
            return np.exp(-(tt / eta) ** c)

        rng = np.random.default_rng(123)
        # Parameter uncertainty → 90% credible band
        shapes = weibull_shape * np.exp(rng.normal(0, 0.08, 400))[:, None]
        scales = weibull_scale * np.exp(rng.normal(0, 0.05, 400))[:, None]
        band_lo, band_hi = np.percentile(weibull_sf(t, shapes, scales), [5, 95], axis=0)
        surv = weibull_sf(t, weibull_shape, weibull_scale)

        # Kaplan–Meier on a simulated, right-censored historical population
        n_hist = 75
        life = weibull_scale * (-np.log(rng.uniform(size=n_hist))) ** (1 / weibull_shape)
        censor = rng.uniform(20, 90, n_hist)
        obs = np.minimum(life, censor)
        event = life <= censor
        order_km = np.argsort(obs)
        km_t, km_s, s, at_risk = [0.0], [1.0], 1.0, n_hist
        for ti, ev in zip(obs[order_km], event[order_km]):
            if ev:
                s *= 1 - 1 / at_risk
                km_t.append(ti)
                km_s.append(s)
            at_risk -= 1
        km_t.append(obs.max())
        km_s.append(s)

        fig_surv = go.Figure()
        fig_surv.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]), y=np.concatenate([band_hi, band_lo[::-1]]),
            fill="toself", fillcolor=rgba(BLUE, 0.13), line=dict(width=0),
            name="90% credible band", hoverinfo="skip",
        ))
        fig_surv.add_trace(go.Scatter(
            x=km_t, y=km_s, mode="lines", line=dict(color=INK, width=1.4, shape="hv"),
            name=f"Kaplan–Meier ({event.sum()} failures / {n_hist})",
            hovertemplate="%{x:.0f} mo · %{y:.0%}<extra></extra>",
        ))
        fig_surv.add_trace(go.Scatter(
            x=t, y=surv, mode="lines", line=dict(color=BLUE, width=2.5), name="Weibull model",
            hovertemplate="%{x:.0f} mo · %{y:.0%}<extra></extra>",
        ))
        for p, label in [(0.10, "B10"), (0.50, "B50")]:
            tb = weibull_scale * (-np.log(1 - p)) ** (1 / weibull_shape)
            fig_surv.add_trace(go.Scatter(
                x=[tb], y=[1 - p], mode="markers+text", showlegend=False,
                marker=dict(size=9, color=NAVY, line=dict(width=1.5, color="white")),
                text=[f"{label} life: {tb:.0f} mo"], textposition="middle right",
                textfont=dict(size=11, color=INK), hoverinfo="skip",
            ))
        # Where the current fleet sits on the curve
        fig_surv.add_trace(go.Scatter(
            x=fleet_meta["age"], y=np.full(FLEET_SIZE, 0.02), mode="markers",
            marker=dict(symbol="line-ns-open", size=12, color=AMBER, line=dict(width=2)),
            name="Current fleet ages", hovertemplate="%{x} mo<extra></extra>",
        ))
        fig_surv.update_xaxes(title_text="Component age (months)", range=[0, 100])
        fig_surv.update_yaxes(title_text="Survival probability", range=[0, 1.04], tickformat=".0%")
        fig_surv.update_layout(legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.99,
                                           bgcolor="rgba(255,255,255,0.9)"))
        chart_card("Population survival curve",
                   f"Weibull β = {weibull_shape:.1f}, η = {weibull_scale} months vs. observed "
                   "Kaplan–Meier estimate", fig_surv, height=420)

    # ── Model Validation: Predicted vs Actual ────────────────────────
    with p_right:
        rng = np.random.default_rng(99)
        actual = rng.uniform(2, 40, 30)
        noise = rng.normal(0, 1, 30)
        predicted = np.clip(actual + noise * 2.8 / np.abs(noise).mean(), 0, 50)   # hybrid model, MAE 2.8
        err = np.abs(predicted - actual)
        pt_colors = [GREEN if e < 3 else AMBER if e < 6 else RED for e in err]

        fig_pa = go.Figure()
        fig_pa.add_trace(go.Scatter(
            x=[0, 45, 45, 0], y=[3, 48, 42, -3], mode="lines", fill="toself", fillcolor=rgba(GREEN, 0.08),
            line=dict(width=0), name="±3 months", hoverinfo="skip",
        ))
        fig_pa.add_trace(go.Scatter(
            x=[0, 45], y=[0, 45], mode="lines", line=dict(color=MUTED, dash="dash", width=1.4),
            name="Perfect prediction", hoverinfo="skip",
        ))
        fig_pa.add_trace(go.Scatter(
            x=actual, y=predicted, mode="markers", showlegend=False,
            marker=dict(size=9, color=pt_colors, line=dict(width=1, color="white")),
            hovertemplate="Actual %{x:.1f} mo<br>Predicted %{y:.1f} mo<extra></extra>",
        ))
        fig_pa.update_xaxes(title_text="Actual RUL (months)", range=[0, 45])
        fig_pa.update_yaxes(title_text="Predicted RUL (months)", range=[0, 48])
        chart_card("Back-test: predicted vs. actual RUL",
                   f"Hybrid model on 30 held-out failures · MAE {err.mean():.1f} months · "
                   f"{np.mean(err < 3):.0%} within ±3 months<br>Error: "
                   f'<span style="color:{GREEN}">● &lt; 3 mo</span> · '
                   f'<span style="color:{AMBER}">● 3–6 mo</span> · '
                   f'<span style="color:{RED}">● &gt; 6 mo</span>', fig_pa, height=420)

    # ── Model Comparison ─────────────────────────────────────────────
    models = ["Weibull baseline", "Random survival forest", "XGBoost",
              "Bayesian hierarchical", "Hybrid (Bayes + ML)"]
    mae_vals = [6.5, 4.1, 3.9, 3.2, 2.8]
    coverage = [None, None, None, 0.89, 0.91]
    fig_cmp = go.Figure(go.Bar(
        y=models, x=mae_vals, orientation="h",
        marker=dict(color=[MUTED, MUTED, MUTED, LIGHT_BLUE, BLUE]),
        text=[f"{v:.1f} mo" + (f"  ·  {c:.0%} PI coverage" if c else "") for v, c in zip(mae_vals, coverage)],
        textposition="outside", textfont=dict(size=11, color=INK), cliponaxis=False,
        hovertemplate="%{y}: MAE %{x:.1f} months<extra></extra>",
    ))
    fig_cmp.update_xaxes(title_text="Mean absolute error (months) — lower is better", range=[0, 9.5])
    fig_cmp.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    fig_cmp.update_layout(margin=dict(t=12))
    chart_card("Model comparison", "Cross-validated mean absolute error on the same back-test set",
               fig_cmp, height=280)


# ══════════════════════════════════════════════════════════════════════
# TAB 4: Methodology
# ══════════════════════════════════════════════════════════════════════
with tab_method:
    mc1, mc2, mc3 = st.columns(3)
    with mc1, st.container(border=True, key="card_method_1"):
        st.markdown("""
        ##### 📥 Data inputs
        - Daily sensor readings (vibration, temperature, pressure, flow)
        - Maintenance event history (failures, repairs, inspections)
        - Operating context (load, duty cycle, plant mode)
        - Component metadata (manufacturer, install date, material)
        """)
    with mc2, st.container(border=True, key="card_method_2"):
        st.markdown("""
        ##### 🧮 Modelling approach
        - Hierarchical Bayesian degradation model with component-level random effects
        - XGBoost feature engineering for health-index construction
        - SHAP-based explainability for degradation driver identification
        - Hybrid ensemble: MAE of 2.8 months with 91% interval coverage
        """)
    with mc3, st.container(border=True, key="card_method_3"):
        st.markdown("""
        ##### 📊 Dashboard outputs
        - Fleet-wide health overview with status classification
        - Per-component RUL posteriors with full uncertainty
        - Degradation projections with confidence fans
        - Replacement windows aligned to planned outages
        """)

    with st.container(border=True, key="card_method_how"):
        st.markdown(f"""
        #### How the numbers on this page are produced
        1. **Health index.** Each pump's observed history follows
           `HI(t) = 100 − mᵢ · (r·t + a·t²) + noise`, where `mᵢ` is the pump's random effect
           (how fast it wears relative to the fleet).
        2. **Projection.** From today's health index, {N_TRAJ} paths per pump are simulated with the
           sidebar's rate `r`, acceleration `a` and noise `σ`, drawing `mᵢ` and `a` from their posteriors.
        3. **RUL.** Each path's RUL is the first time it crosses the failure threshold; the
           distribution of those times is the RUL posterior. Paths that never cross within
           {HORIZON} months are right-censored at {HORIZON}.
        4. **Status.** Median RUL < {CRITICAL_MO} months → *Critical*; < {WATCH_MO} months →
           *Watch*; otherwise *Healthy*.
        5. **Planning.** A pump is assigned the latest planned outage it reaches with ≥ 90% probability
           (its 10th-percentile RUL). If none qualifies, replacement is expedited.
        """)

st.markdown("---")
st.caption("RUL Prediction Dashboard · Proof of concept · All data is synthetic · Built with Streamlit & Plotly")
