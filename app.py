# ============================================================
# BSDI AGENTIC AI — MAIN STREAMLIT APP
# Theme: Deep forest green + gold
# ============================================================

import os
import base64
import socket
from pathlib import Path
from urllib.parse import urlparse
from html import escape

import streamlit as st
import pandas as pd

from src.ingestion.excel_loader import load_projects
from src.graph.track_a import run_track_a_traced
from src.graph.track_b import run_track_b_detailed
from src.agents.coordinator_agent import (
    run_coordinator,
    make_project_decision,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BSDI AI",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATA
# ============================================================

DATA_PATH = Path("data/Projects.xlsx")


@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/Projects.xlsx not found."
        )

    data = load_projects(DATA_PATH)

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    # Numeric columns
    if "Cost (M)" in data.columns:
        data["Cost (M)"] = pd.to_numeric(
            data["Cost (M)"],
            errors="coerce",
        ).fillna(0)

    if "Progress %" in data.columns:
        data["Progress %"] = pd.to_numeric(
            data["Progress %"],
            errors="coerce",
        )

    # Clean text columns
    for column in [
        "District",
        "Category",
        "Status",
        "Contractor",
        "XEN Name",
        "Executing Agency",
    ]:

        if column in data.columns:

            data[column] = (
                data[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

            data.loc[
                data[column] == "",
                column,
            ] = "Unknown"

    return data


try:

    df = load_data()

except Exception as error:

    st.error(
        f"Could not load Projects.xlsx: {error}"
    )

    st.stop()


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "qwen3:4b",
)

parsed_ollama = urlparse(OLLAMA_URL)


@st.cache_data(ttl=15)
def ollama_is_reachable(
    host=None,
    port=None,
):

    host = host or parsed_ollama.hostname or "localhost"
    port = port or parsed_ollama.port or 11434

    try:

        with socket.create_connection(
            (host, port),
            timeout=1,
        ):
            return True

    except OSError:

        return False


OLLAMA_UP = ollama_is_reachable()


# ============================================================
# DESIGN TOKENS
# ============================================================

GREEN_DEEP = "#0d2a1a"
GREEN_MID = "#14432b"
GREEN_BASE = "#1c5c3a"
GREEN_LIGHT = "#e8f3ec"

GOLD = "#e8b93f"
GOLD_DARK = "#c99a2e"

MINT = "#6ee7b7"

INK = "#132018"
MUTED = "#5b6b62"

PAPER = "#f7f9f7"
CARD_BORDER = "#dfe7e1"

RED = "#c1443a"
AMBER = "#c9822e"


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    f"""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap'
);


/* ----------------------------------------------------------
   GLOBAL
---------------------------------------------------------- */

html,
body,
[class*="css"] {{
    font-family:
        'Inter',
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}}

.stApp {{
    background: {PAPER};
}}

.block-container {{
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 4rem;
}}

h1,
h2,
h3,
h4,
.hero-title,
.brand-title {{
    font-family: 'Poppins', sans-serif;
}}


/* ----------------------------------------------------------
   SIDEBAR
---------------------------------------------------------- */

section[data-testid="stSidebar"] {{
    background: {GREEN_DEEP};
    border-right: 1px solid #0a1f13;
}}

section[data-testid="stSidebar"] * {{
    color: #eef5f0;
}}

section[data-testid="stSidebar"] label span {{
    color: #d7e6dc !important;
}}

section[data-testid="stSidebar"]
[role="radiogroup"]
label:hover {{
    background: rgba(232,185,63,0.08);
    border-radius: 8px;
}}


/* ----------------------------------------------------------
   BRAND BAR
---------------------------------------------------------- */

.brand-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;

    background: #ffffff;

    border: 1px solid {CARD_BORDER};
    border-radius: 16px;

    padding: 16px 26px;
    margin-bottom: 22px;

    box-shadow:
        0 4px 18px rgba(20,67,43,0.06);
}}

.brand-left {{
    display: flex;
    align-items: center;
    gap: 16px;
}}

.brand-badge {{
    width: 52px;
    height: 52px;

    border-radius: 50%;

    background:
        linear-gradient(
            160deg,
            {GREEN_MID},
            {GREEN_DEEP}
        );

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 24px;

    box-shadow:
        inset 0 0 0 2px {GOLD};
}}

.brand-title {{
    color: {GREEN_BASE};
    font-size: 21px;
    font-weight: 800;
    letter-spacing: .2px;
    line-height: 1.1;
}}

.brand-subtitle {{
    color: {MUTED};
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-top: 2px;
}}

.brand-pill {{
    background: {GREEN_LIGHT};
    color: {GREEN_BASE};

    border: 1px solid #cfe4d8;
    border-radius: 20px;

    padding: 7px 14px;

    font-size: 12px;
    font-weight: 700;
}}


/* ----------------------------------------------------------
   HERO
---------------------------------------------------------- */

.hero {{
    position: relative;
    overflow: hidden;

    background:
        radial-gradient(
            circle at 78% 30%,
            rgba(110,231,183,0.16),
            transparent 55%
        ),
        linear-gradient(
            155deg,
            {GREEN_MID} 0%,
            {GREEN_DEEP} 100%
        );

    border: 1px solid #0a2417;
    border-radius: 20px;

    padding: 40px 44px;
    margin-bottom: 26px;

    box-shadow:
        0 20px 50px rgba(10,30,18,.28);
}}

.hero-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 8px;

    background:
        rgba(232,185,63,0.12);

    border:
        1px solid rgba(232,185,63,0.35);

    color: {GOLD};

    border-radius: 20px;

    padding: 7px 16px;

    font-size: 12.5px;
    font-weight: 700;

    margin-bottom: 18px;
}}

.hero-title {{
    color: #ffffff;

    font-size: 40px;
    font-weight: 800;

    letter-spacing: -0.8px;
    line-height: 1.15;

    max-width: 720px;
}}

.hero-title .accent {{
    color: {GOLD};
}}

.hero-subtitle {{
    color: #cfe0d5;

    font-size: 15px;
    line-height: 1.7;

    margin-top: 14px;

    max-width: 620px;
}}

.online {{
    display: inline-flex;
    align-items: center;
    gap: 7px;

    margin-top: 22px;

    padding: 8px 15px;

    border-radius: 20px;

    background:
        rgba(110,231,183,0.12);

    border:
        1px solid rgba(110,231,183,0.4);

    color: {MINT};

    font-size: 12px;
    font-weight: 700;
}}

.online.offline {{
    background:
        rgba(193,68,58,0.14);

    border:
        1px solid rgba(193,68,58,0.4);

    color: #ff9d92;
}}

.online .dot {{
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: {MINT};

    box-shadow:
        0 0 0 3px rgba(110,231,183,0.25);
}}

.online.offline .dot {{
    background: #ff5c4d;

    box-shadow:
        0 0 0 3px rgba(255,92,77,0.25);
}}


/* ----------------------------------------------------------
   SECTIONS
---------------------------------------------------------- */

.section-title {{
    color: {GREEN_BASE};

    font-size: 22px;
    font-weight: 800;

    margin-top: 28px;
    margin-bottom: 6px;
}}

.section-subtitle {{
    color: {MUTED};

    font-size: 14px;

    margin-bottom: 18px;
}}


/* ----------------------------------------------------------
   METRIC CARDS
---------------------------------------------------------- */

.metric-card {{
    background: #ffffff;

    border: 1px solid {CARD_BORDER};
    border-left: 4px solid {GOLD};

    border-radius: 14px;

    padding: 20px 22px;

    min-height: 122px;

    box-shadow:
        0 6px 18px rgba(20,67,43,.05);

    transition:
        transform .18s ease,
        box-shadow .18s ease;
}}

.metric-card:hover {{
    transform: translateY(-3px);

    box-shadow:
        0 14px 28px rgba(20,67,43,.12);
}}

.metric-card.mint {{
    border-left-color: {MINT};
}}

.metric-label {{
    color: {MUTED};

    font-size: 11.5px;
    font-weight: 700;

    letter-spacing: .5px;
    text-transform: uppercase;
}}

.metric-value {{
    color: {INK};

    font-size: 28px;
    font-weight: 800;

    margin-top: 8px;

    font-family: 'Poppins', sans-serif;
}}

.metric-description {{
    color: #8a9a90;

    font-size: 12px;

    margin-top: 4px;
}}


/* ----------------------------------------------------------
   AGENT CARDS
---------------------------------------------------------- */

.agent-card {{
    background: #ffffff;

    border: 1px solid {CARD_BORDER};

    border-radius: 16px;

    padding: 22px;

    min-height: 156px;

    text-align: center;

    transition:
        transform .2s ease,
        box-shadow .2s ease,
        border-color .2s ease;
}}

.agent-card:hover {{
    transform: translateY(-4px);

    box-shadow:
        0 16px 32px rgba(20,67,43,.12);

    border-color: {GOLD};
}}

.agent-icon {{
    width: 46px;
    height: 46px;

    border-radius: 50%;

    background: {GREEN_LIGHT};

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 22px;

    margin: 0 auto;
}}

.agent-title {{
    color: {INK};

    font-size: 16px;
    font-weight: 700;

    margin-top: 12px;

    font-family: 'Poppins', sans-serif;
}}

.agent-description {{
    color: {MUTED};

    font-size: 12.5px;
    line-height: 1.5;

    margin-top: 7px;
}}

.agent-status {{
    display: inline-block;

    color: #1a8a53;
    background: {GREEN_LIGHT};

    border-radius: 12px;

    padding: 3px 10px;

    font-size: 11px;
    font-weight: 700;

    margin-top: 14px;
}}


/* ----------------------------------------------------------
   RESULT CARD
---------------------------------------------------------- */

.result-card {{
    background: #ffffff;

    border: 1px solid {CARD_BORDER};
    border-left: 5px solid {GREEN_BASE};

    border-radius: 16px;

    padding: 26px 28px;

    margin-top: 18px;

    box-shadow:
        0 8px 24px rgba(20,67,43,.07);
}}

.result-header {{
    display: flex;
    align-items: center;
    gap: 10px;

    color: {GREEN_BASE};

    font-size: 18px;
    font-weight: 800;

    font-family: 'Poppins', sans-serif;

    margin-bottom: 20px;
}}

.result-check {{
    display: inline-flex;

    align-items: center;
    justify-content: center;

    width: 28px;
    height: 28px;

    border-radius: 50%;

    background: {GREEN_LIGHT};

    color: {GREEN_BASE};

    font-size: 15px;
}}

.result-label {{
    color: {MUTED};

    font-size: 11px;
    font-weight: 800;

    letter-spacing: .8px;
    text-transform: uppercase;
}}

.result-question {{
    color: {INK};

    font-size: 15px;
    font-weight: 600;

    line-height: 1.6;

    margin-top: 6px;
    margin-bottom: 20px;
}}

.result-answer {{
    color: {INK};

    font-size: 16px;

    line-height: 1.75;

    background: {GREEN_LIGHT};

    border-radius: 12px;

    padding: 16px 18px;

    margin-top: 8px;

    white-space: pre-wrap;
    word-wrap: break-word;
}}


/* ----------------------------------------------------------
   CHART CONTAINERS
---------------------------------------------------------- */

.chart-card {{
    background: #ffffff;

    border: 1px solid {CARD_BORDER};

    border-radius: 16px;

    padding: 18px 20px 10px;

    box-shadow:
        0 6px 18px rgba(20,67,43,.05);

    min-height: 410px;
}}

.chart-title {{
    color: {INK};

    font-family: 'Poppins', sans-serif;

    font-size: 15px;
    font-weight: 700;

    margin-bottom: 10px;
}}


/* ----------------------------------------------------------
   BADGES
---------------------------------------------------------- */

.chip {{
    display: inline-block;

    background: {GREEN_LIGHT};
    color: {GREEN_BASE};

    border: 1px solid #cfe4d8;

    border-radius: 20px;

    padding: 4px 12px;

    font-size: 12px;
    font-weight: 700;

    margin: 2px 4px 2px 0;
}}

.risk-high {{
    background: rgba(193,68,58,0.10);
    color: {RED};

    border:
        1px solid rgba(193,68,58,0.35);

    border-radius: 20px;

    padding: 3px 12px;

    font-size: 11.5px;
    font-weight: 800;
}}

.risk-medium {{
    background: rgba(201,130,46,0.12);
    color: {AMBER};

    border:
        1px solid rgba(201,130,46,0.35);

    border-radius: 20px;

    padding: 3px 12px;

    font-size: 11.5px;
    font-weight: 800;
}}

.decision-fund {{
    background: rgba(26,138,83,0.10);
    color: #1a8a53;

    border:
        1px solid rgba(26,138,83,0.35);

    border-radius: 20px;

    padding: 3px 12px;

    font-size: 11.5px;
    font-weight: 800;
}}

.decision-hold {{
    background: rgba(201,130,46,0.12);
    color: {AMBER};

    border:
        1px solid rgba(201,130,46,0.35);

    border-radius: 20px;

    padding: 3px 12px;

    font-size: 11.5px;
    font-weight: 800;
}}

.decision-review {{
    background: rgba(91,107,98,0.12);
    color: {MUTED};

    border:
        1px solid rgba(91,107,98,0.3);

    border-radius: 20px;

    padding: 3px 12px;

    font-size: 11.5px;
    font-weight: 800;
}}


/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

.stButton > button {{
    min-height: 46px;

    border-radius: 10px;

    font-weight: 700;

    background: {GREEN_BASE} !important;

    border:
        1px solid {GREEN_BASE} !important;

    color: #ffffff !important;

    transition:
        transform .15s ease,
        box-shadow .15s ease;
}}

.stButton > button:hover {{
    transform: translateY(-1px);

    box-shadow:
        0 10px 22px rgba(28,92,58,.28);

    background: {GREEN_MID} !important;
}}


/* ----------------------------------------------------------
   INPUTS
---------------------------------------------------------- */

.stTextArea textarea,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] {{
    border-radius: 10px !important;
}}


/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.footer {{
    border-top: 1px solid {CARD_BORDER};

    margin-top: 50px;

    padding-top: 20px;

    text-align: center;

    color: #8a9a90;

    font-size: 12px;
}}

</style>
"""
)


# ============================================================
# HELPERS
# ============================================================

def risk_badge(level):

    css = (
        "risk-high"
        if str(level).upper() == "HIGH"
        else "risk-medium"
    )

    return (
        f'<span class="{css}">'
        f'{escape(str(level))}'
        f'</span>'
    )


def decision_badge(decision):

    value = str(decision).upper()

    if value == "FUND":
        css = "decision-fund"

    elif "HOLD" in value:
        css = "decision-hold"

    else:
        css = "decision-review"

    return (
        f'<span class="{css}">'
        f'{escape(str(decision))}'
        f'</span>'
    )


def ollama_warning():

    st.warning(
        f"⚠️ Can't reach Ollama at `{OLLAMA_URL}`. "
        f"The AI tracks require the `{OLLAMA_MODEL}` model. "
        "The Dashboard and Projects Explorer still work."
    )


def render_metric_card(
    label,
    value,
    description,
    mint=False,
):

    css = "metric-card mint" if mint else "metric-card"

    st.html(
        f"""
        <div class="{css}">
            <div class="metric-label">
                {escape(str(label))}
            </div>

            <div class="metric-value">
                {escape(str(value))}
            </div>

            <div class="metric-description">
                {escape(str(description))}
            </div>
        </div>
        """
    )


def render_result_card(
    question,
    answer,
    steps=0,
):

    safe_question = escape(
        str(question)
    )

    safe_answer = escape(
        str(answer)
    )

    st.html(
        f"""
        <div class="result-card">

            <div class="result-header">
                <span class="result-check">✓</span>
                Analysis Complete
            </div>

            <div class="result-label">
                Your Question
            </div>

            <div class="result-question">
                {safe_question}
            </div>

            <div class="result-label">
                Answer
            </div>

            <div class="result-answer">
                {safe_answer}
            </div>

        </div>
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    logo_path = Path("assets/bsdi_logo.jpeg")
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8") if logo_path.exists() else ""

    st.html(
        f"""
        <div style="
            font-size:24px;
            font-weight:800;
            color:#ffffff;
            margin-bottom:2px;
            font-family:'Poppins',sans-serif;
            display:flex;
            align-items:center;
            gap:8px;
        ">
            {f'<img src="data:image/jpeg;base64,{logo_b64}" style="width:42px;height:42px;object-fit:contain;vertical-align:middle;">' if logo_b64 else ''}
            BSDI AI
        </div>

        <div style="
            color:{MINT};
            font-size:10.5px;
            font-weight:700;
            letter-spacing:1px;
            text-transform:uppercase;
            margin-bottom:26px;
        ">
            Infrastructure Intelligence
        </div>
        """
    )

    st.markdown(
        "### Navigation"
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "💬 Data Assistant",
            "⚠️ Risk Audit",
            "🤖 Review Board",
            "📊 Projects Explorer",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    if OLLAMA_UP:

        st.html(
            f"""
            <div style="
                color:{MINT};
                font-size:12px;
                font-weight:700;
            ">
                ● Ollama Online
            </div>

            <div style="
                color:#9db3a5;
                font-size:11px;
                margin-top:5px;
            ">
                Local open-source LLM
                ({escape(OLLAMA_MODEL)})
            </div>
            """
        )

    else:

        st.html(
            """
            <div style="
                color:#ff9d92;
                font-size:12px;
                font-weight:700;
            ">
                ● Ollama Not Reachable
            </div>

            <div style="
                color:#9db3a5;
                font-size:11px;
                margin-top:5px;
            ">
                Start Ollama to use AI tracks.
            </div>
            """
        )


# ============================================================
# TOP BRAND BAR
# ============================================================

st.html(
    f"""
    <div class="brand-bar">

        <div class="brand-left">

            <div class="brand-badge">
                🏔️
            </div>

            <div>

                <div class="brand-title">
                    BSDI
                </div>

                <div class="brand-subtitle">
                    Program Implementation Unit
                </div>

            </div>

        </div>

        <div class="brand-pill">
            Government of Balochistan
        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

districts_count_all = (
    df["District"].nunique()
    if "District" in df.columns
    else 0
)

online_class = (
    "online"
    if OLLAMA_UP
    else "online offline"
)

online_label = (
    "Local AI System Online"
    if OLLAMA_UP
    else "Local AI System Offline"
)

st.html(
    f"""
    <div class="hero">

        <div class="hero-eyebrow">
            ⭐ Transforming Balochistan
        </div>

        <div class="hero-title">
            Building Tomorrow's
            <span class="accent">
                Balochistan
            </span>
            AI
        </div>

        <div class="hero-subtitle">
            AI-powered infrastructure analysis,
            risk auditing and multi-agent project
            funding decisions — across all
            {districts_count_all}
            districts represented in the dataset.
        </div>

        <div class="{online_class}">
            <span class="dot"></span>
            {online_label}
        </div>

    </div>
    """
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

total_projects = len(df)

total_cost = (
    df["Cost (M)"].sum()
    if "Cost (M)" in df.columns
    else 0
)

completed_count = (
    int(
        df["Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("completed")
        .sum()
    )
    if "Status" in df.columns
    else 0
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.html(
        """
        <div class="section-title">
            Project Overview
        </div>

        <div class="section-subtitle">
            Live overview of the BSDI infrastructure
            project dataset.
        </div>
        """
    )


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    filter_col1, filter_col2, filter_col3 = (
        st.columns(3)
    )

    dashboard_df = df.copy()


    with filter_col1:

        if "District" in df.columns:

            district_options = sorted(
                df["District"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            dashboard_district = st.selectbox(
                "District",
                ["All"] + district_options,
                key="dashboard_district",
            )

            if dashboard_district != "All":

                dashboard_df = dashboard_df[
                    dashboard_df["District"]
                    .astype(str)
                    .str.strip()
                    .eq(dashboard_district)
                ]


    with filter_col2:

        if "Category" in df.columns:

            category_options = sorted(
                df["Category"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            dashboard_category = st.selectbox(
                "Category",
                ["All"] + category_options,
                key="dashboard_category",
            )

            if dashboard_category != "All":

                dashboard_df = dashboard_df[
                    dashboard_df["Category"]
                    .astype(str)
                    .str.strip()
                    .eq(dashboard_category)
                ]


    with filter_col3:

        if "Status" in df.columns:

            status_options = sorted(
                df["Status"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            dashboard_status = st.selectbox(
                "Status",
                ["All"] + status_options,
                key="dashboard_status",
            )

            if dashboard_status != "All":

                dashboard_df = dashboard_df[
                    dashboard_df["Status"]
                    .astype(str)
                    .str.strip()
                    .eq(dashboard_status)
                ]


    # --------------------------------------------------------
    # FILTERED METRICS
    # --------------------------------------------------------

    dashboard_total_projects = len(
        dashboard_df
    )

    dashboard_total_cost = (
        dashboard_df["Cost (M)"].sum()
        if "Cost (M)" in dashboard_df.columns
        else 0
    )

    dashboard_not_started = (
        int(
            dashboard_df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("not started")
            .sum()
        )
        if "Status" in dashboard_df.columns
        else 0
    )

    dashboard_districts = (
        dashboard_df["District"].nunique()
        if "District" in dashboard_df.columns
        else 0
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        render_metric_card(
            "TOTAL PROJECTS",
            f"{dashboard_total_projects:,}",
            "Projects in BSDI dataset",
        )

    with c2:

        render_metric_card(
            "TOTAL PROJECT VALUE",
            f"{dashboard_total_cost:,.1f}",
            "Million PKR",
            mint=True,
        )

    with c3:

        render_metric_card(
            "NOT STARTED",
            f"{dashboard_not_started:,}",
            "Projects awaiting work",
        )

    with c4:

        render_metric_card(
            "DISTRICTS",
            f"{dashboard_districts:,}",
            "Covered by current filter",
            mint=True,
        )


    # --------------------------------------------------------
    # AI REVIEW SYSTEM
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-title">
            AI Review System
        </div>

        <div class="section-subtitle">
            Multiple AI agents analyze the same
            infrastructure dataset.
        </div>
        """
    )

    a1, a2, a3 = st.columns(3)


    with a1:

        st.html(
            """
            <div class="agent-card">

                <div class="agent-icon">
                    💬
                </div>

                <div class="agent-title">
                    Track A — Data Assistant
                </div>

                <div class="agent-description">
                    Ask questions about projects,
                    districts, categories and costs.
                </div>

                <div class="agent-status">
                    ● READY
                </div>

            </div>
            """
        )


    with a2:

        st.html(
            """
            <div class="agent-card">

                <div class="agent-icon">
                    ⚠️
                </div>

                <div class="agent-title">
                    Track B — Risk Audit
                </div>

                <div class="agent-description">
                    Identify delivery, financial and
                    accountability risks.
                </div>

                <div class="agent-status">
                    ● READY
                </div>

            </div>
            """
        )


    with a3:

        st.html(
            """
            <div class="agent-card">

                <div class="agent-icon">
                    🤖
                </div>

                <div class="agent-title">
                    Track C — Review Board
                </div>

                <div class="agent-description">
                    Finance, Delivery and Equity agents
                    collaborate on funding.
                </div>

                <div class="agent-status">
                    ● READY
                </div>

            </div>
            """
        )


    # ============================================================
# TRACK A — DATA ASSISTANT
# ============================================================

elif page == "💬 Data Assistant":

    st.html(
        """
        <div class="section-title">
            Track A — Data Assistant
        </div>

        <div class="section-subtitle">
            Ask questions about the BSDI project dataset
            using natural language. The agent uses the
            real project data to answer.
        </div>
        """
    )


    if not OLLAMA_UP:

        ollama_warning()


    if "track_a_history" not in st.session_state:

        st.session_state.track_a_history = []


    # --------------------------------------------------------
    # EXAMPLE QUESTIONS
    # --------------------------------------------------------

    example_questions = [
        "How many water projects in Kech are completed?",
        "What's the total budget of all Not Started projects?",
        "Which district has the most education projects?",
        "List the 5 most expensive health projects.",
    ]


    st.caption(
        "Try an example or type your own question."
    )


    example_cols = st.columns(
        len(example_questions)
    )

    example_clicked = None


    for index, (
        col,
        example,
    ) in enumerate(
        zip(
            example_cols,
            example_questions,
        )
    ):

        with col:

            if st.button(
                example,
                key=f"example_{index}",
                width="stretch",
            ):

                example_clicked = example


    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = st.text_area(
        "Your question",
        value=(
            example_clicked
            if example_clicked
            else ""
        ),
        placeholder=(
            "Example: Which district has "
            "the most projects?"
        ),
        height=100,
    )


    run_clicked = st.button(
        "🔎 Run Analysis",
        width="stretch",
        type="primary",
    )


    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    if run_clicked or example_clicked:

        active_question = (
            question
            or example_clicked
            or ""
        ).strip()


        if not active_question:

            st.warning(
                "Please enter a question."
            )


        elif not OLLAMA_UP:

            st.error(
                "Ollama is not reachable. "
                "Start Ollama and try again."
            )


        else:

            with st.spinner(
                "Analyzing BSDI data..."
            ):

                try:

                    result = run_track_a_traced(
                        active_question
                    )

                    st.session_state.track_a_history.insert(
                        0,
                        result,
                    )

                    st.success(
                        "Analysis complete."
                    )

                except Exception as error:

                    st.error(
                        f"Track A error: {error}"
                    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    for index, item in enumerate(
        st.session_state.track_a_history
    ):

        render_result_card(
            question=item.get(
                "question",
                "",
            ),
            answer=item.get(
                "answer",
                "",
            ),
            steps=item.get(
                "steps",
                0,
            ),
        )




        if index < len(
            st.session_state.track_a_history
        ) - 1:

            st.divider()


# ============================================================
# TRACK B — RISK AUDIT
# ============================================================

elif page == "⚠️ Risk Audit":

    st.html(
        """
        <div class="section-title">
            Track B — Risk Audit
        </div>

        <div class="section-subtitle">
            The agent plans its own checks for the stated
            goal, runs each check against the dataset and
            ranks the findings.
        </div>
        """
    )


    if not OLLAMA_UP:

        ollama_warning()


    audit_goal = st.text_area(
        "Audit goal",
        value=(
            "Find the projects most at risk of "
            "failing or being mismanaged."
        ),
        height=100,
    )


    if st.button(
        "⚠️ Run Risk Audit",
        width="stretch",
        type="primary",
    ):

        if not audit_goal.strip():

            st.warning(
                "Please enter an audit goal."
            )


        elif not OLLAMA_UP:

            st.error(
                "Ollama is not reachable. "
                "Start Ollama and try again."
            )


        else:

            with st.spinner(
                "Planning checks and auditing the portfolio..."
            ):

                try:

                    st.session_state.track_b_result = (
                        run_track_b_detailed(
                            audit_goal.strip()
                        )
                    )

                    st.success(
                        "Risk audit complete."
                    )

                except Exception as error:

                    st.error(
                        f"Track B error: {error}"
                    )


    result = st.session_state.get(
        "track_b_result"
    )


    if result:

        # ----------------------------------------------------
        # FINDINGS
        # ----------------------------------------------------

        st.markdown(
            "### 2. Findings"
        )


        for check_result in result.get(
            "results",
            [],
        ):

            count = check_result.get(
                "count",
                0,
            )

            level = check_result.get(
                "risk_level",
                "MEDIUM",
            )

            score = check_result.get(
                "priority_score",
                0,
            )

            findings = check_result.get(
                "findings",
                [],
            )


            with st.expander(
                f"{check_result.get('check', 'Check')} "
                f"— {count} affected",
                expanded=(
                    str(level).upper()
                    == "HIGH"
                ),
            ):

                st.html(
                    f"""
                    {risk_badge(level)}
                    &nbsp;&nbsp;
                    <span class="chip">
                        priority score {escape(str(score))}
                    </span>
                    """
                )


                st.write(
                    check_result.get(
                        "description",
                        "",
                    )
                )


                if (
                    "threshold_m_pkr"
                    in check_result
                    and check_result[
                        "threshold_m_pkr"
                    ] is not None
                ):

                    st.caption(
                        "Threshold: ≥ "
                        f"{check_result['threshold_m_pkr']:.1f}"
                        " M PKR"
                    )


                if (
                    "threshold_percent"
                    in check_result
                ):

                    st.caption(
                        "Threshold: ≥ "
                        f"{check_result['threshold_percent']}"
                        "% of district budget "
                        "Not Started"
                    )


                if findings:

                    findings_df = pd.DataFrame(
                        findings
                    )

                    st.dataframe(
                        findings_df,
                        width="stretch",
                        height=min(
                            320,
                            60
                            + 35
                            * len(findings_df),
                        ),
                    )

                else:

                    st.caption(
                        "No affected rows found."
                    )


        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        st.markdown(
            "### 3. Ranked risk report"
        )


        report = result.get(
            "report",
            "",
        )


        st.html(
            f"""
            <div class="result-card">

                <div class="result-header">
                    <span class="result-check">
                        ⚠
                    </span>

                    Risk Assessment
                </div>

                <div class="result-answer">
                    {escape(str(report))}
                </div>

            </div>
            """
        )


# ============================================================
# TRACK C — REVIEW BOARD
# ============================================================

elif page == "🤖 Review Board":

    st.html(
        """
        <div class="section-title">
            Track C — Multi-Agent Review Board
        </div>

        <div class="section-subtitle">
            Finance, Delivery and Equity agents independently
            review the Not Started portfolio. The Coordinator
            merges their findings, resolves conflicts and
            produces a ranked funding shortlist.
        </div>
        """
    )


    st.info(
        "Funding scenario: an extra PKR 2 billion is "
        "available. Which Not Started projects should "
        "be funded first?"
    )


    if st.button(
        "🤖 Run Review Board",
        width="stretch",
        type="primary",
    ):

        if not OLLAMA_UP:

            st.error(
                "Ollama is not reachable. "
                "Start Ollama and try again."
            )

        else:

            with st.spinner(
                "Finance, Delivery and Equity agents are reviewing..."
            ):

                try:

                    st.session_state.track_c_result = (
                        run_coordinator()
                    )

                    st.success(
                        "Board review complete."
                    )

                except Exception as error:

                    st.error(
                        f"Review Board error: {error}"
                    )


    result = st.session_state.get(
        "track_c_result"
    )


    if result:

        # ----------------------------------------------------
        # SPECIALIST AGENTS
        # ----------------------------------------------------

        st.markdown(
            "### 1. Specialist agent findings"
        )


        f1, f2, f3 = st.columns(3)


        finance = result.get(
            "finance",
            {},
        )

        delivery = result.get(
            "delivery",
            {},
        )

        equity = result.get(
            "equity",
            {},
        )


        with f1:

            st.html(
                f"""
                <div class="agent-card"
                     style="text-align:left;">

                    <div class="agent-title">
                        💰 Finance Agent
                    </div>

                    <div class="agent-description">

                        {len(
                            finance.get(
                                "recommendations",
                                [],
                            )
                        )}
                        cost-efficient candidates flagged.

                        <br>

                        {len(
                            finance.get(
                                "concerns",
                                [],
                            )
                        )}
                        cost concern(s) raised.

                        <br><br>

                        Unfunded pipeline:
                        <b>
                        {finance.get(
                            "evidence",
                            {}
                        ).get(
                            "total_not_started_budget_m_pkr",
                            0
                        ):,.1f}
                        M PKR
                        </b>

                        vs

                        {finance.get(
                            "evidence",
                            {}
                        ).get(
                            "funding_limit_m_pkr",
                            0
                        ):,.1f}
                        M PKR envelope.

                    </div>

                </div>
                """
            )


        with f2:

            st.html(
                f"""
                <div class="agent-card"
                     style="text-align:left;">

                    <div class="agent-title">
                        🚧 Delivery Agent
                    </div>

                    <div class="agent-description">

                        {len(
                            delivery.get(
                                "recommendations",
                                [],
                            )
                        )}
                        project(s) ready with no
                        missing accountability fields.

                        <br>

                        {len(
                            delivery.get(
                                "concerns",
                                [],
                            )
                        )}
                        project(s) missing contractor /
                        XEN information.

                    </div>

                </div>
                """
            )


        with f3:

            st.html(
                f"""
                <div class="agent-card"
                     style="text-align:left;">

                    <div class="agent-title">
                        ⚖️ Equity Agent
                    </div>

                    <div class="agent-description">

                        {len(
                            equity.get(
                                "least_funded_districts",
                                [],
                            )
                        )}
                        district(s) flagged as
                        under-allocated.

                        <br>

                        {len(
                            equity.get(
                                "least_funded_categories",
                                [],
                            )
                        )}
                        category/categories flagged
                        as under-allocated.

                    </div>

                </div>
                """
            )


        # ----------------------------------------------------
        # TRADE-OFFS
        # ----------------------------------------------------

        st.markdown(
            "### 2. Trade-offs the Coordinator resolved"
        )


        tradeoffs = result.get(
            "tradeoffs",
            [],
        )


        if tradeoffs:

            for tradeoff in tradeoffs:

                project_name = tradeoff.get(
                    "project",
                    "Unknown project",
                )

                tradeoff_type = tradeoff.get(
                    "type",
                    "Trade-off",
                )


                with st.expander(
                    f"{project_name} — {tradeoff_type}"
                ):

                    st.markdown(
                        "**Finance says:** "
                        + str(
                            tradeoff.get(
                                "finance",
                                "",
                            )
                        )
                    )

                    st.markdown(
                        "**Delivery says:** "
                        + str(
                            tradeoff.get(
                                "delivery",
                                "",
                            )
                        )
                    )

                    st.markdown(
                        "**Equity says:** "
                        + str(
                            tradeoff.get(
                                "equity",
                                "",
                            )
                        )
                    )

                    st.markdown(
                        "**Coordinator's resolution:** "
                        + str(
                            tradeoff.get(
                                "resolution",
                                "",
                            )
                        )
                    )

        else:

            st.caption(
                "No direct conflicts were detected "
                "between agents this run."
            )


        # ----------------------------------------------------
        # RANKED SHORTLIST
        # ----------------------------------------------------

        st.markdown(
            "### 3. Ranked shortlist & funding decision"
        )


        selected = result.get(
            "selected",
            [],
        )


        shortlist_rows = []


        for project in selected:

            decision, reason = (
                make_project_decision(
                    project
                )
            )


            shortlist_rows.append(
                {
                    "Global ID": project.get(
                        "global_id"
                    ),

                    "District": project.get(
                        "district"
                    ),

                    "Category": project.get(
                        "category"
                    ),

                    "Cost (M PKR)": project.get(
                        "cost_m_pkr"
                    ),

                    "Finance Support":
                        "✔"
                        if project.get(
                            "finance_support"
                        )
                        else "—",

                    "Equity Support":
                        "✔"
                        if project.get(
                            "equity_support"
                        )
                        else "—",

                    "Delivery Warning":
                        "⚠"
                        if project.get(
                            "delivery_warning"
                        )
                        else "—",

                    "Decision": decision,
                }
            )


        if shortlist_rows:

            st.dataframe(
                pd.DataFrame(
                    shortlist_rows
                ),
                width="stretch",
                hide_index=True,
            )

        else:

            st.caption(
                "No candidates were selected."
            )


        # ----------------------------------------------------
        # FUNDING METRICS
        # ----------------------------------------------------

        b1, b2 = st.columns(2)


        with b1:

            st.metric(
                "Total recommended funding (M PKR)",
                f"{result.get('total_funding', 0):,.1f}",
            )


        with b2:

            st.metric(
                "Remaining envelope (M PKR)",
                f"{result.get('remaining_funding', 0):,.1f}",
            )


        # ----------------------------------------------------
        # FINAL RECOMMENDATION
        # ----------------------------------------------------

        st.markdown(
            "### 4. Final recommendation"
        )


        recommendation = result.get(
            "recommendation",
            "",
        )


        st.html(
            f"""
            <div class="result-card">

                <div class="result-header">
                    <span class="result-check">
                        ✓
                    </span>

                    Board Recommendation
                </div>

                <div class="result-answer">
                    {escape(
                        str(recommendation)
                    )}
                </div>

            </div>
            """
        )


# ============================================================
# PROJECTS EXPLORER
# ============================================================

elif page == "📊 Projects Explorer":

    st.html(
        """
        <div class="section-title">
            Projects Explorer
        </div>

        <div class="section-subtitle">
            Search, filter and inspect individual
            BSDI projects.
        </div>
        """
    )


    filtered = df.copy()


    f1, f2, f3 = st.columns(3)


    # --------------------------------------------------------
    # DISTRICT FILTER
    # --------------------------------------------------------

    with f1:

        if "District" in df.columns:

            district_options = sorted(
                df["District"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            district = st.selectbox(
                "District",
                ["All"] + district_options,
                key="explorer_district",
            )

            if district != "All":

                filtered = filtered[
                    filtered["District"]
                    .astype(str)
                    .eq(district)
                ]


    # --------------------------------------------------------
    # CATEGORY FILTER
    # --------------------------------------------------------

    with f2:

        if "Category" in df.columns:

            category_options = sorted(
                df["Category"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            category = st.selectbox(
                "Category",
                ["All"] + category_options,
                key="explorer_category",
            )

            if category != "All":

                filtered = filtered[
                    filtered["Category"]
                    .astype(str)
                    .eq(category)
                ]


    # --------------------------------------------------------
    # STATUS FILTER
    # --------------------------------------------------------

    with f3:

        if "Status" in df.columns:

            status_options = sorted(
                df["Status"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            status = st.selectbox(
                "Status",
                ["All"] + status_options,
                key="explorer_status",
            )

            if status != "All":

                filtered = filtered[
                    filtered["Status"]
                    .astype(str)
                    .eq(status)
                ]


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_text = st.text_input(
        "Search description / contractor / Global ID",
        "",
    )


    if search_text.strip():

        needle = (
            search_text
            .strip()
            .lower()
        )

        search_columns = [
            column
            for column in [
                "Description",
                "Contractor",
                "Global ID",
            ]
            if column in filtered.columns
        ]


        if search_columns:

            mask = pd.Series(
                False,
                index=filtered.index,
            )


            for column in search_columns:

                mask = (
                    mask
                    |
                    filtered[column]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        needle,
                        na=False,
                    )
                )


            filtered = filtered[mask]


    st.html(
        f"""
        <div style="
            color:{GREEN_BASE};
            font-weight:700;
            font-size:14px;
            margin-bottom:10px;
        ">
            {len(filtered):,}
            projects found
        </div>
        """
    )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    preferred_columns = [
        "Global ID",
        "District",
        "Category",
        "Description",
        "Cost (M)",
        "Progress %",
        "Status",
        "Executing Agency",
        "Contractor",
        "XEN Name",
        "NITs",
    ]


    display_columns = [
        column
        for column in preferred_columns
        if column in filtered.columns
    ]


    st.dataframe(
        filtered[display_columns],
        width="stretch",
        height=500,
        hide_index=True,
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "⬇️ Download filtered results (CSV)",

        data=(
            filtered[
                display_columns
            ]
            .to_csv(index=False)
            .encode("utf-8")
        ),

        file_name=(
            "bsdi_projects_filtered.csv"
        ),

        mime="text/csv",
    )


    # --------------------------------------------------------
    # PROJECT DETAILS
    # --------------------------------------------------------

    st.markdown(
        "### Project Details"
    )


    if "Global ID" in filtered.columns:

        project_ids = (
            filtered["Global ID"]
            .dropna()
            .astype(str)
            .tolist()
        )


        if project_ids:

            selected_id = st.selectbox(
                "Select a project",
                project_ids,
                key="selected_project",
            )


            selected = filtered[
                filtered["Global ID"]
                .astype(str)
                .eq(selected_id)
            ]


            if not selected.empty:

                project = (
                    selected
                    .iloc[0]
                    .to_dict()
                )


                left, right = st.columns(2)


                with left:

                    for key in [
                        "Global ID",
                        "District",
                        "Category",
                        "Status",
                        "Progress %",
                        "Cost (M)",
                    ]:

                        if key in project:

                            st.write(
                                f"**{key}:** "
                                f"{project[key]}"
                            )


                with right:

                    for key in [
                        "Executing Agency",
                        "Contractor",
                        "XEN Name",
                        "XEN Contact",
                        "NITs",
                        "Work Started",
                    ]:

                        if key in project:

                            st.write(
                                f"**{key}:** "
                                f"{project[key]}"
                            )


                if "Description" in project:

                    st.markdown(
                        "#### Description"
                    )

                    st.write(
                        project["Description"]
                    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    f"""
    <div class="footer">
        BSDI Agentic AI · Qwen3 + Ollama + LangGraph
        · Local Open-Source LLM Architecture
    </div>
    """
)