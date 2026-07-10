"""
dashboard.py

Interactive Streamlit dashboard for the Advance Analytics Intelligence
Platform POC. Reads directly from adv_platform.duckdb.

Pages: Overview, Time Trend, Pipeline, CRM Engagement, Financials, Risk
Scoring — filterable by Relationship Manager and a Start/End date pair.

Interactivity:
  - Overview KPI cards have a "View Trend" button that jumps to the Time
    Trend page pre-set to that metric.
  - The "Clients Flagged Red" KPI jumps to Risk Scoring pre-filtered to RED.
  - The Risk Distribution chart is clickable (click a RED/AMBER/GREEN bar)
    and also has fallback quick-filter buttons; either shows a client-level
    table (RM, Client Name, Ending Assets, Revenue) at the bottom of Overview.

Run:
    streamlit run dashboard.py

By default it looks for adv_platform.duckdb in the same folder as this
script. If your database lives elsewhere, either:
  - copy/symlink it next to this script, or
  - edit DB_PATH below, or
  - set an environment variable before running:
        export ADV_PLATFORM_DB=/path/to/adv_platform.duckdb   (WSL/Ubuntu)
        $env:ADV_PLATFORM_DB="C:\\path\\to\\adv_platform.duckdb"  (PowerShell)
"""

import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = os.environ.get("ADV_PLATFORM_DB", "adv_platform.duckdb")

st.set_page_config(
    page_title="Advance Analytics Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

# ----------------------------------------------------------------------
# Responsive styling — tighter padding and wrapping metrics on narrow
# (mobile/tablet) viewports. Streamlit's columns already stack under
# ~640px automatically; this just tidies things up at that breakpoint.
# ----------------------------------------------------------------------
st.markdown("""
<style>
@media (max-width: 640px) {
    div[data-testid="stMetric"] { padding: 0.4rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    h1 { font-size: 1.5rem !important; }
    h2, h3 { font-size: 1.15rem !important; }
}
div[data-testid="stMetric"] {
    background-color: rgba(128,128,128,0.06);
    border-radius: 8px;
    padding: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Connection + cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def get_connection():
    db_path = Path(DB_PATH)
    if not db_path.exists():
        st.error(
            f"Could not find '{db_path}'. Place adv_platform.duckdb next to "
            f"dashboard.py, or set the ADV_PLATFORM_DB environment variable "
            f"to its full path."
        )
        st.stop()
    return duckdb.connect(str(db_path), read_only=True)


def rm_clause(rms, alias=""):
    prefix = f"{alias}." if alias else ""
    if not rms:
        return ""
    return "AND {}relationship_manager IN ({})".format(
        prefix, ",".join(f"'{r}'" for r in rms)
    )


@st.cache_data
def load_rm_list(_con):
    return _con.execute(
        "SELECT DISTINCT relationship_manager FROM policy_financials ORDER BY 1"
    ).df()["relationship_manager"].tolist()


@st.cache_data
def load_financials(_con, rms, start_date, end_date):
    return _con.execute(f"""
        SELECT *
        FROM policy_financials
        WHERE week_start_date BETWEEN '{start_date}' AND '{end_date}'
        {rm_clause(rms)}
    """).df()


@st.cache_data
def load_crm(_con, rms, start_date, end_date):
    return _con.execute(f"""
        SELECT *
        FROM crm_interactions
        WHERE interaction_date BETWEEN '{start_date}' AND '{end_date}'
        {rm_clause(rms)}
    """).df()


@st.cache_data
def load_stage_velocity(_con, rms):
    return _con.execute(f"""
        WITH stage_transitions AS (
            SELECT
                opportunity_id, client_id, relationship_manager,
                sales_stage, sequence, transition_date, expected_aum,
                LEAD(transition_date) OVER (
                    PARTITION BY opportunity_id ORDER BY sequence
                ) AS next_transition_date
            FROM opportunities_pipeline
            WHERE 1=1 {rm_clause(rms)}
        )
        SELECT
            sales_stage,
            COUNT(*) AS transitions,
            ROUND(AVG(DATE_DIFF('day', transition_date, next_transition_date)), 1) AS avg_days_in_stage,
            SUM(expected_aum) AS total_expected_aum
        FROM stage_transitions
        WHERE next_transition_date IS NOT NULL
        GROUP BY sales_stage
        ORDER BY avg_days_in_stage DESC
    """).df()


@st.cache_data
def load_stalled_opps(_con, rms):
    return _con.execute(f"""
        WITH latest_stage AS (
            SELECT
                opportunity_id, client_id, relationship_manager, sales_stage,
                transition_date, expected_aum,
                ROW_NUMBER() OVER (PARTITION BY opportunity_id ORDER BY sequence DESC) AS rn
            FROM opportunities_pipeline
            WHERE 1=1 {rm_clause(rms)}
        )
        SELECT
            opportunity_id, client_id, relationship_manager, sales_stage,
            transition_date AS entered_current_stage,
            DATE_DIFF('day', transition_date, CURRENT_DATE) AS days_in_current_stage,
            expected_aum
        FROM latest_stage
        WHERE rn = 1 AND sales_stage NOT IN ('Win-100%', 'Loss-0%')
        ORDER BY days_in_current_stage DESC
        LIMIT 25
    """).df()


@st.cache_data
def load_win_loss_ratio(_con, rms):
    """
    Count-based win/loss ratio per the deck's Chapter 4 definition:
    wins / (wins + losses), counted as distinct OpportunityIDs that
    reached Win-100% or Loss-0%. Note this is scoped by the RM filter
    only, not the sidebar date range — it reflects the full pipeline
    history, same as the other Pipeline-tab queries (stage velocity,
    stalled opportunities).
    """
    row = _con.execute(f"""
        WITH counts AS (
            SELECT
                COUNT(DISTINCT opportunity_id) FILTER (WHERE sales_stage = 'Win-100%') AS wins,
                COUNT(DISTINCT opportunity_id) FILTER (WHERE sales_stage = 'Loss-0%') AS losses
            FROM opportunities_pipeline
            WHERE 1=1 {rm_clause(rms)}
        )
        SELECT
            wins, losses,
            CASE WHEN (wins + losses) > 0
                 THEN ROUND(wins * 100.0 / (wins + losses), 1)
                 ELSE 0
            END AS win_ratio_pct
        FROM counts
    """).fetchone()
    return {"wins": row[0], "losses": row[1], "win_ratio_pct": row[2]}


@st.cache_data
def load_risk_scores(_con, rms):
    return _con.execute(f"""
        WITH as_of AS (
            SELECT GREATEST(MAX(week_start_date), MAX(d)) AS as_of_date
            FROM policy_financials, (SELECT MAX(interaction_date) AS d FROM crm_interactions)
        ),
        recent_fin AS (
            SELECT
                client_id, relationship_manager,
                AVG(revenue) FILTER (
                    WHERE week_start_date > (SELECT as_of_date FROM as_of) - INTERVAL 56 DAY
                ) AS revenue_recent,
                AVG(revenue) FILTER (
                    WHERE week_start_date <= (SELECT as_of_date FROM as_of) - INTERVAL 56 DAY
                      AND week_start_date > (SELECT as_of_date FROM as_of) - INTERVAL 112 DAY
                ) AS revenue_prior,
                AVG(ending_assets) FILTER (
                    WHERE week_start_date > (SELECT as_of_date FROM as_of) - INTERVAL 56 DAY
                ) AS assets_recent,
                AVG(ending_assets) FILTER (
                    WHERE week_start_date <= (SELECT as_of_date FROM as_of) - INTERVAL 56 DAY
                      AND week_start_date > (SELECT as_of_date FROM as_of) - INTERVAL 112 DAY
                ) AS assets_prior,
                SUM(net_flows) FILTER (
                    WHERE week_start_date > (SELECT as_of_date FROM as_of) - INTERVAL 56 DAY
                ) AS net_flows_recent
            FROM policy_financials
            GROUP BY client_id, relationship_manager
        ),
        latest_point AS (
            SELECT DISTINCT ON (client_id)
                client_id, ending_assets AS latest_ending_assets, revenue AS latest_revenue
            FROM policy_financials
            ORDER BY client_id, week_start_date DESC
        ),
        last_touch AS (
            SELECT client_id, MAX(interaction_date) AS last_interaction_date,
                   AVG(engagement_score) AS avg_engagement_score
            FROM crm_interactions GROUP BY client_id
        ),
        open_pipeline AS (
            SELECT client_id,
                   COUNT(DISTINCT opportunity_id) FILTER (
                       WHERE sales_stage NOT IN ('Win-100%', 'Loss-0%')
                   ) AS open_opportunities,
                   COUNT(DISTINCT opportunity_id) FILTER (
                       WHERE sales_stage = 'Loss-0%'
                   ) AS lost_opportunities
            FROM opportunities_pipeline GROUP BY client_id
        ),
        scored AS (
            SELECT
                f.client_id, f.relationship_manager,
                d.client_name,
                lp.latest_ending_assets, lp.latest_revenue,
                t.last_interaction_date,
                DATE_DIFF('day', t.last_interaction_date, (SELECT as_of_date FROM as_of)) AS days_since_touch,
                ROUND(t.avg_engagement_score, 1) AS avg_engagement_score,
                COALESCE(p.open_opportunities, 0) AS open_opportunities,
                LEAST(40,
                    CASE WHEN f.revenue_prior > 0 AND f.revenue_recent < f.revenue_prior * 0.95 THEN 15 ELSE 0 END
                    + CASE WHEN f.assets_prior > 0 AND f.assets_recent < f.assets_prior * 0.97 THEN 15 ELSE 0 END
                    + CASE WHEN f.net_flows_recent < 0 THEN 10 ELSE 0 END
                ) AS financial_health_points,
                LEAST(35,
                    CASE
                        WHEN t.last_interaction_date IS NULL THEN 20
                        WHEN DATE_DIFF('day', t.last_interaction_date, (SELECT as_of_date FROM as_of)) > 21 THEN 20
                        WHEN DATE_DIFF('day', t.last_interaction_date, (SELECT as_of_date FROM as_of)) > 10 THEN 10
                        ELSE 0
                    END + CASE WHEN t.avg_engagement_score < 5 THEN 15 ELSE 0 END
                ) AS relationship_points,
                LEAST(25,
                    CASE WHEN COALESCE(p.open_opportunities, 0) = 0 THEN 20 ELSE 0 END
                    + CASE WHEN COALESCE(p.lost_opportunities, 0) > 0 THEN 5 ELSE 0 END
                ) AS pipeline_points
            FROM recent_fin f
            LEFT JOIN dim_clients d USING (client_id)
            LEFT JOIN latest_point lp USING (client_id)
            LEFT JOIN last_touch t USING (client_id)
            LEFT JOIN open_pipeline p USING (client_id)
            WHERE 1=1 {rm_clause(rms, "f")}
        ),
        ranked AS (
            SELECT *,
                financial_health_points + relationship_points + pipeline_points AS risk_score,
                -- NTILE(10) buckets ROWS as evenly as possible into 10 groups,
                -- regardless of how many clients share the same score. This
                -- matters because the point system is coarse (few discrete
                -- outcomes per dimension), so many clients tie on the same
                -- total score. PERCENT_RANK grades ties by VALUE, so a large
                -- tied cluster can jump straight past the 10-30% Amber
                -- window (or land entirely inside/outside it) depending on
                -- how big the clusters above it happen to be — meaning
                -- Amber could silently vanish for the full book while still
                -- appearing once filtered to a smaller RM subset that
                -- clusters differently. NTILE avoids that: it always
                -- allocates ~10% of ROWS to decile 1 and ~20% to deciles
                -- 2-3, so Amber is populated consistently at any filter size.
                NTILE(10) OVER (
                    ORDER BY financial_health_points + relationship_points + pipeline_points DESC, client_id
                ) AS score_decile
            FROM scored
        )
        SELECT
            client_id, client_name, relationship_manager, risk_score,
            CASE
                WHEN score_decile <= 1 THEN 'RED'
                WHEN score_decile <= 3 THEN 'AMBER'
                ELSE 'GREEN'
            END AS rag_flag,
            financial_health_points, relationship_points, pipeline_points,
            days_since_touch, avg_engagement_score, open_opportunities,
            latest_ending_assets, latest_revenue
        FROM ranked
        ORDER BY risk_score DESC
    """).df()


# Single source of truth for RAG colors/emoji, used by the chart, the
# quick-filter buttons, and the styled table — previously the chart used
# orange (#ff7f0e) for Amber while the filter button used a yellow emoji,
# which looked inconsistent. Everything now pulls from this one dict.
RAG_COLORS = {
    "RED":   {"main": "#d62728", "light": "#f8d7da", "emoji": "🔴"},
    "AMBER": {"main": "#ff7f0e", "light": "#ffe8cc", "emoji": "🟠"},
    "GREEN": {"main": "#2ca02c", "light": "#d4edda", "emoji": "🟢"},
}

TREND_METRICS = {
    "ending_assets": {
        "label": "Ending Assets",
        # Point-in-time: last observed value within each period, not an
        # average or sum, since assets are a balance/stock metric.
        "agg": "ARG_MAX(ending_assets, week_start_date)",
        "prefix": "$", "is_percent": False,
    },
    "revenue": {"label": "Revenue", "agg": "SUM(revenue)", "prefix": "$", "is_percent": False},
    "net_flows": {"label": "Net Flows", "agg": "SUM(net_flows)", "prefix": "$", "is_percent": False},
    "revenue_vs_goal": {
        "label": "Revenue vs Goal (%)",
        "agg": "ROUND(SUM(revenue) / NULLIF(SUM(revenue_goal), 0) * 100, 1)",
        "prefix": "", "is_percent": True,
    },
}
GRANULARITY_TO_TRUNC = {"Yearly": "year", "Quarterly": "quarter", "Monthly": "month", "Weekly": "week"}


@st.cache_data
def load_trend(_con, rms, start_date, end_date, granularity, metric, split_by_rm):
    trunc_unit = GRANULARITY_TO_TRUNC[granularity]
    agg_expr = TREND_METRICS[metric]["agg"]
    group_cols = "period" + (", relationship_manager" if split_by_rm else "")
    select_rm = ", relationship_manager" if split_by_rm else ""
    return _con.execute(f"""
        SELECT
            DATE_TRUNC('{trunc_unit}', week_start_date) AS period,
            {agg_expr} AS value
            {select_rm}
        FROM policy_financials
        WHERE week_start_date BETWEEN '{start_date}' AND '{end_date}'
        {rm_clause(rms)}
        GROUP BY {group_cols}
        ORDER BY period
    """).df()


@st.cache_data
def load_point_in_time_assets(_con, rms, as_of_date):
    """
    Ending Assets is a stock/balance metric, not a flow — it should never
    be summed across weeks. This returns the sum, across selected
    clients, of each client's most recent ending_assets snapshot on or
    before as_of_date (i.e. the sidebar's End Date).
    """
    return _con.execute(f"""
        SELECT SUM(ending_assets) AS total_ending_assets
        FROM (
            SELECT DISTINCT ON (client_id) client_id, ending_assets
            FROM policy_financials
            WHERE week_start_date <= '{as_of_date}'
            {rm_clause(rms)}
            ORDER BY client_id, week_start_date DESC
        )
    """).fetchone()[0] or 0


@st.cache_data
def load_trend_client_detail(_con, rms, start_date, end_date, granularity, metric):
    """
    Client-level breakdown behind the Time Trend chart: one row per
    period per client, with Relationship Manager and Client Name, for
    the tabular view. Note this can be a large table (period buckets x
    number of clients) — Streamlit's dataframe widget handles this fine
    since it virtualizes rendering, but it's worth knowing at Weekly
    granularity with no RM filter this may be several thousand rows.
    """
    trunc_unit = GRANULARITY_TO_TRUNC[granularity]
    agg_expr = TREND_METRICS[metric]["agg"]
    return _con.execute(f"""
        SELECT
            DATE_TRUNC('{trunc_unit}', f.week_start_date) AS period,
            f.relationship_manager,
            COALESCE(d.client_name, f.client_id) AS client_name,
            {agg_expr} AS value
        FROM policy_financials f
        LEFT JOIN dim_clients d ON d.client_id = f.client_id
        WHERE f.week_start_date BETWEEN '{start_date}' AND '{end_date}'
        {rm_clause(rms, "f")}
        GROUP BY period, f.relationship_manager, COALESCE(d.client_name, f.client_id)
        ORDER BY period, f.relationship_manager, client_name
    """).df()


@st.cache_data
def load_rag_detail(_con, rms, rag_flag):
    risk_df = load_risk_scores(_con, rms)
    subset = risk_df[risk_df["rag_flag"] == rag_flag].copy()
    subset["latest_ending_assets"] = subset["latest_ending_assets"].apply(lambda v: f"${v:,.0f}")
    subset["latest_revenue"] = subset["latest_revenue"].apply(lambda v: f"${v:,.1f}")
    return subset[["relationship_manager", "client_name", "latest_ending_assets", "latest_revenue"]].rename(
        columns={
            "relationship_manager": "Relationship Manager",
            "client_name": "Client Name",
            "latest_ending_assets": "Ending Assets",
            "latest_revenue": "Revenue",
        }
    )


# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"
if "trend_metric" not in st.session_state:
    st.session_state.trend_metric = "ending_assets"
if "risk_rag_filter" not in st.session_state:
    st.session_state.risk_rag_filter = None
if "overview_selected_rag" not in st.session_state:
    st.session_state.overview_selected_rag = None


def set_nav(page, **state_updates):
    """
    Callback used as a button's on_click. Callbacks run BEFORE the script
    body re-renders, so it's safe to set session_state for a key that's
    bound to a widget (like the sidebar radio's key="nav_page") here —
    doing the same assignment inline in the script body would raise
    StreamlitAPIException because that widget has already been
    instantiated earlier in the same run.
    """
    st.session_state.nav_page = page
    for k, v in state_updates.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
con = get_connection()

st.sidebar.title("📊 Adv Analytics Platform")
st.sidebar.caption("POC — Advance Analytics Intelligence Platform")

all_rms = load_rm_list(con)
selected_rms = st.sidebar.multiselect(
    "Relationship Manager(s)", options=all_rms, default=[],
    help="Leave empty to include all RMs"
)

min_date, max_date = con.execute(
    "SELECT MIN(week_start_date), MAX(week_start_date) FROM policy_financials"
).fetchone()

st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input(
    "Start Date", value=min_date, min_value=min_date, max_value=max_date, key="start_date_input"
)
end_date = st.sidebar.date_input(
    "End Date", value=max_date, min_value=min_date, max_value=max_date, key="end_date_input"
)
if start_date > end_date:
    st.sidebar.error("Start Date must be on or before End Date.")
    st.stop()

st.sidebar.divider()
nav_options = ["Overview", "Time Trend", "Pipeline", "CRM Engagement", "Financials", "Risk Scoring"]
st.sidebar.radio("Go to", nav_options, key="nav_page")
st.sidebar.divider()
st.sidebar.caption(f"Connected to: `{Path(DB_PATH).name}`")


# ----------------------------------------------------------------------
# Load filtered data
# ----------------------------------------------------------------------
fin = load_financials(con, selected_rms, start_date, end_date)
crm = load_crm(con, selected_rms, start_date, end_date)
stage_vel = load_stage_velocity(con, selected_rms)
stalled = load_stalled_opps(con, selected_rms)
risk = load_risk_scores(con, selected_rms)
win_loss = load_win_loss_ratio(con, selected_rms)

st.title("Advance Analytics Intelligence Platform")
st.caption("Interactive POC dashboard · synthetic data · read-only DuckDB connection")

page = st.session_state.nav_page

# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------
if page == "Overview":
    total_assets = load_point_in_time_assets(con, selected_rms, end_date)
    total_revenue = fin["revenue"].sum() if not fin.empty else 0
    total_net_flows = fin["net_flows"].sum() if not fin.empty else 0
    goal_attainment = (
        (fin["revenue"].sum() / fin["revenue_goal"].sum() * 100)
        if not fin.empty and fin["revenue_goal"].sum() > 0 else 0
    )
    red_count = (risk["rag_flag"] == "RED").sum() if not risk.empty else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.caption(f"As of {end_date}")
        st.metric("Total Ending Assets", f"${total_assets/1e9:,.0f}B")
        st.button("View Trend →", key="btn_assets", width='stretch',
                  on_click=set_nav, kwargs={"page": "Time Trend", "trend_metric": "ending_assets"})
    with col2:
        st.caption(f"As of {start_date} to {end_date}")
        st.metric("Total Revenue", f"${total_revenue/1e6:,.1f}M")
        st.button("View Trend →", key="btn_revenue", width='stretch',
                  on_click=set_nav, kwargs={"page": "Time Trend", "trend_metric": "revenue"})
    with col3:
        st.caption(f"As of {start_date} to {end_date}")
        st.metric("Revenue vs Goal", f"{goal_attainment:,.1f}%")
        st.button("View Trend →", key="btn_goal", width='stretch',
                  on_click=set_nav, kwargs={"page": "Time Trend", "trend_metric": "revenue_vs_goal"})
    with col4:
        st.caption(f"As of {start_date} to {end_date}")
        st.metric("Clients Flagged Red", int(red_count))
        st.button("View Red Clients →", key="btn_red", width='stretch',
                  on_click=set_nav, kwargs={"page": "Risk Scoring", "risk_rag_filter": "RED"})
    with col5:
        st.caption(f"As of {start_date} to {end_date}")
        st.metric("Net Flows", f"${total_net_flows/1e6:,.1f}M")
        st.button("View Trend →", key="btn_netflows", width='stretch',
                  on_click=set_nav, kwargs={"page": "Time Trend", "trend_metric": "net_flows"})
    with col6:
        st.caption("All time (not date-filtered)")
        st.metric("Win/Loss Ratio", f"{win_loss['win_ratio_pct']:.1f}%")
        st.button("View Pipeline →", key="btn_winloss", width='stretch',
                  on_click=set_nav, kwargs={"page": "Pipeline"})

    st.subheader("Risk distribution across selected book")
    st.caption("Click a bar, or use the buttons below, to see the clients behind it.")
    if not risk.empty:
        rag_counts = risk["rag_flag"].value_counts().reindex(
            ["RED", "AMBER", "GREEN"]
        ).fillna(0).reset_index()
        rag_counts.columns = ["rag_flag", "clients"]
        fig = px.bar(
            rag_counts, x="rag_flag", y="clients", color="rag_flag",
            color_discrete_map={k: v["main"] for k, v in RAG_COLORS.items()},
            text="clients",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
        selection = st.plotly_chart(
            fig, width='stretch', on_select="rerun",
            selection_mode="points", key="rag_chart",
        )

        clicked_rag = None
        try:
            points = selection.selection.points if hasattr(selection, "selection") else selection["selection"]["points"]
            if points:
                clicked_rag = points[0].get("x")
        except Exception:
            clicked_rag = None
        if clicked_rag:
            st.session_state.overview_selected_rag = clicked_rag

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        if bcol1.button(f"{RAG_COLORS['RED']['emoji']} Red clients", width='stretch'):
            st.session_state.overview_selected_rag = "RED"
        if bcol2.button(f"{RAG_COLORS['AMBER']['emoji']} Amber clients", width='stretch'):
            st.session_state.overview_selected_rag = "AMBER"
        if bcol3.button(f"{RAG_COLORS['GREEN']['emoji']} Green clients", width='stretch'):
            st.session_state.overview_selected_rag = "GREEN"
        if bcol4.button("Clear selection", width='stretch'):
            st.session_state.overview_selected_rag = None

        if st.session_state.overview_selected_rag:
            sel = st.session_state.overview_selected_rag
            st.subheader(f"Clients flagged {sel}")
            detail = load_rag_detail(con, selected_rms, sel).reset_index(drop=True)
            detail.insert(0, "S. No", range(1, len(detail) + 1))
            st.dataframe(detail, width='stretch', hide_index=True)
    else:
        st.info("No data for the current filter selection.")

# ----------------------------------------------------------------------
# Time Trend
# ----------------------------------------------------------------------
elif page == "Time Trend":
    st.subheader("Time Trend")

    tcol1, tcol2, tcol3 = st.columns([1, 1, 1])
    with tcol1:
        granularity = st.radio("Granularity", list(GRANULARITY_TO_TRUNC.keys()), horizontal=True, index=2)
    with tcol2:
        metric_labels = {v["label"]: k for k, v in TREND_METRICS.items()}
        default_label = TREND_METRICS[st.session_state.trend_metric]["label"]
        chosen_label = st.selectbox(
            "Metric", list(metric_labels.keys()),
            index=list(metric_labels.keys()).index(default_label),
        )
        metric_key = metric_labels[chosen_label]
    with tcol3:
        split_by_rm = st.checkbox("Split by Relationship Manager", value=False)

    trend_df = load_trend(con, selected_rms, start_date, end_date, granularity, metric_key, split_by_rm)

    if trend_df.empty:
        st.info("No data for the current filter selection.")
    else:
        is_percent = TREND_METRICS[metric_key]["is_percent"]
        trend_df = trend_df.sort_values("period").reset_index(drop=True)

        def make_period_label(df):
            # One label format per granularity, reused for the chart's
            # x-axis and both tables' Period columns so they always match.
            if granularity == "Yearly":
                return df["period"].dt.strftime("%Y")
            elif granularity == "Quarterly":
                return "Q" + df["period"].dt.quarter.astype(str) + " " + df["period"].dt.year.astype(str)
            elif granularity == "Monthly":
                return df["period"].dt.strftime("%b %Y")
            else:  # Weekly
                return df["period"].dt.strftime("%b %-d, %Y")

        def format_value(v):
            if is_percent:
                return f"{v:.1f}%"
            elif metric_key == "ending_assets":
                return f"${v:,.0f}"
            else:
                return f"${v:,.1f}"

        trend_df["period_label"] = make_period_label(trend_df)
        trend_df["value_label"] = trend_df["value"].apply(format_value)

        x_col = "period_label"
        category_orders = {x_col: trend_df[x_col].tolist()}

        # --- Chart type: line for Weekly or any % metric, bar otherwise ---
        use_line = (granularity == "Weekly") or is_percent

        plot_kwargs = {"labels": {"value": chosen_label, x_col: granularity}, "category_orders": category_orders}
        if split_by_rm:
            plot_kwargs["color"] = "relationship_manager"

        if use_line:
            fig = px.line(trend_df, x=x_col, y="value", markers=True, **plot_kwargs)
        else:
            fig = px.bar(trend_df, x=x_col, y="value", text="value_label", **plot_kwargs)
            fig.update_traces(textposition="outside")
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            if split_by_rm:
                fig.update_layout(barmode="group")

        if is_percent:
            fig.update_yaxes(ticksuffix="%", tickformat=",.0f")
            fig.update_traces(hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>")
        elif metric_key == "ending_assets":
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            fig.update_traces(hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>")
        else:
            fig.update_yaxes(tickprefix="$", tickformat=",.1f")
            fig.update_traces(hovertemplate="%{x}<br>$%{y:,.1f}<extra></extra>")

        st.plotly_chart(fig, width='stretch')

        # --- Tabular view: client-level detail, Period formatted to match
        # the chart's x-axis exactly, plus Relationship Manager and Client
        # Name for each row.
        st.subheader("Client-level detail")
        detail_df = load_trend_client_detail(con, selected_rms, start_date, end_date, granularity, metric_key)
        if detail_df.empty:
            st.info("No client-level detail for the current filter selection.")
        else:
            detail_df["period"] = pd.to_datetime(detail_df["period"])
            detail_df = detail_df.sort_values(
                ["period", "relationship_manager", "client_name"]
            ).reset_index(drop=True)
            detail_df["Period"] = make_period_label(detail_df)
            detail_df[chosen_label] = detail_df["value"].apply(format_value)
            display_df = detail_df[["Period", "relationship_manager", "client_name", chosen_label]].rename(
                columns={"relationship_manager": "Relationship Manager", "client_name": "Client Name"}
            )
            st.dataframe(display_df, width='stretch', hide_index=True)

# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
elif page == "Pipeline":
    st.subheader("Average days spent per sales stage")
    if not stage_vel.empty:
        fig = px.bar(
            stage_vel, x="sales_stage", y="avg_days_in_stage",
            text="avg_days_in_stage", labels={"avg_days_in_stage": "Avg days"},
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No pipeline transitions for the current filter selection.")

    st.subheader("Top stalled opportunities (open, longest in current stage)")
    st.dataframe(stalled, width='stretch', hide_index=True)

# ----------------------------------------------------------------------
# CRM Engagement
# ----------------------------------------------------------------------
elif page == "CRM Engagement":
    st.subheader("Interaction volume by type")
    if not crm.empty:
        type_counts = crm["interaction_type"].value_counts().reset_index()
        type_counts.columns = ["interaction_type", "count"]
        fig = px.pie(type_counts, names="interaction_type", values="count")
        st.plotly_chart(fig, width='stretch')

        st.subheader("Engagement score trend over time")
        trend = crm.groupby("interaction_date")["engagement_score"].mean().reset_index()
        fig2 = px.line(trend, x="interaction_date", y="engagement_score")
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No CRM interactions for the current filter selection.")

# ----------------------------------------------------------------------
# Financials
# ----------------------------------------------------------------------
elif page == "Financials":
    st.subheader("Revenue vs. Revenue Goal over time")
    if not fin.empty:
        weekly = fin.groupby("week_start_date")[["revenue", "revenue_goal"]].sum().reset_index()
        fig = px.line(weekly, x="week_start_date", y=["revenue", "revenue_goal"])
        st.plotly_chart(fig, width='stretch')

        st.subheader("Ending assets over time")
        assets_trend = fin.groupby("week_start_date")["ending_assets"].sum().reset_index()
        fig2 = px.area(assets_trend, x="week_start_date", y="ending_assets")
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No financials data for the current filter selection.")

# ----------------------------------------------------------------------
# Risk Scoring
# ----------------------------------------------------------------------
elif page == "Risk Scoring":
    if st.session_state.risk_rag_filter:
        st.info(f"Filtered to {st.session_state.risk_rag_filter} clients (from Overview). "
                f"Use the selector below to change or clear this.")
    rag_options = ["All", "RED", "AMBER", "GREEN"]
    default_idx = rag_options.index(st.session_state.risk_rag_filter) if st.session_state.risk_rag_filter in rag_options else 0
    rag_pick = st.selectbox("Filter by RAG flag", rag_options, index=default_idx)
    st.session_state.risk_rag_filter = None if rag_pick == "All" else rag_pick

    display_risk = risk if rag_pick == "All" else risk[risk["rag_flag"] == rag_pick]

    st.subheader(f"Highest-risk clients ({len(display_risk)} shown)")
    rag_colors = {k: f"background-color:{v['light']}" for k, v in RAG_COLORS.items()}
    styler = display_risk.head(25).style
    color_fn = lambda v: rag_colors.get(v, "")
    styler = styler.map(color_fn, subset=["rag_flag"]) if hasattr(styler, "map") \
        else styler.applymap(color_fn, subset=["rag_flag"])
    st.dataframe(styler, width='stretch', hide_index=True)

    st.subheader("Risk score breakdown for a specific client")
    if not display_risk.empty:
        selected_client = st.selectbox("Select ClientID", display_risk["client_id"].tolist())
        row = display_risk[display_risk["client_id"] == selected_client].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Financial Health pts (/40)", int(row["financial_health_points"]))
        c2.metric("Relationship pts (/35)", int(row["relationship_points"]))
        c3.metric("Pipeline pts (/25)", int(row["pipeline_points"]))
        st.metric("Total Risk Score", f"{int(row['risk_score'])} — {row['rag_flag']}")