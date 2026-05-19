import io
import logging
import random
from datetime import date, timedelta
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd                   # top-level - no more deferred import inside build_csv
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml

# ==============================================================================
# LOCAL / PROJECT IMPORTS  (kept deferred inside cached loaders to avoid
# circular-import issues at Streamlit startup - listed here for visibility)
#   data.physics_utils              - IntraDayNormStats, DayFeatureNormStats
#   solar_diffusion.vae             - SolarVAE
#   solar_diffusion.denoiser        - SolarDenoiser
#   solar_diffusion.optical_depth   - LatentTauTransform, TauNoiseSchedule
#   solar_diffusion.generate        - generate_sequence
#   scipy.stats                     - gaussian_kde  (inside plot_statistics)
# ==============================================================================

# APP METADATA
APP_VERSION = "v1.0.0"

# FILE PATHS  -  every path lives here; nowhere else in this file
REPO_ROOT      = Path(__file__).parent
CONFIG_PATH    = REPO_ROOT / "config.yaml"
VAE_PATH       = REPO_ROOT / "inference_bundle" / "vae_weights.pt"
DIFF_PATH      = REPO_ROOT / "inference_bundle" / "denoiser_weights.pt"
TAU_PATH       = REPO_ROOT / "inference_bundle" / "latent_tau_stats.json"
INTRADAY_STATS  = REPO_ROOT / "data" / "norm_intraday.json"
DAY_FEAT_STATS  = REPO_ROOT / "data" / "norm_day_feat.json"
CLIMATE_STATS   = REPO_ROOT / "data" / "norm_climate_stats.json"
GMM_PATH       = REPO_ROOT / "data" / "regime_gmm.json"   # needed by generate.py for class_frequencies

# Utils physics should be also in the data folder 

# All required files - used for startup sanity check
REQUIRED_FILES: list = [
    CONFIG_PATH,
    VAE_PATH,
    DIFF_PATH,
    TAU_PATH,
    INTRADAY_STATS,
    DAY_FEAT_STATS,
    # GMM is optional — generate.py falls back gracefully if absent, but
    # the location prior will use uniform cloudy sub-class split instead of
    # real training frequencies. Warn rather than hard-fail if missing.
]
OPTIONAL_FILES: list = [GMM_PATH]

# INPUT VALIDATION CONSTANTS  - single source of truth for all bounds
LAT_MIN,  LAT_MAX  = -90.0,   90.0
LON_MIN,  LON_MAX  = -180.0, 180.0
DATE_MIN           = date(2000,  1,  1)
DATE_MAX           = date(2100, 12, 31)
MAX_DAYS           = 365    # hard cap - refuses generation beyond this
WARN_DAYS          = 180    # shows performance warning above this

# MATPLOTLIB / LOGGING SETUP
plt.rcParams.update({
    "figure.dpi":          200,
    "savefig.dpi":         200,
    "figure.autolayout":   False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "font.family":         "monospace",
    "text.color":          "#f0f4f8",
    "axes.labelcolor":     "#9db4c8",
    "xtick.color":         "#f0f4f8",
    "ytick.color":         "#f0f4f8",
    "path.simplify":       False,
    "lines.antialiased":   True,
    "patch.antialiased":   True,
    "text.antialiased":    True,
    "figure.facecolor":    "none",
    "axes.facecolor":      "none",
    "savefig.facecolor":   "none",
    "savefig.transparent": True,
})

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# STATION METADATA
STATION_META = {
    "BON": {"lat": 40.05192,  "lon": -88.37309,  "name": "Bondville, IL",          "climate": "Humid continental"},
    "DRA": {"lat": 36.62373,  "lon": -116.01947, "name": "Desert Rock, NV",        "climate": "Hot desert"},
    "FPK": {"lat": 48.30783,  "lon": -105.10170, "name": "Fort Peck, MT",          "climate": "Semi-arid, cold"},
    "GWN": {"lat": 34.25470,  "lon": -89.87290,  "name": "Goodwin Creek, MS",      "climate": "Humid subtropical"},
    "PSU": {"lat": 40.72012,  "lon": -77.93085,  "name": "Penn State, PA",         "climate": "Humid continental"},
    "SXF": {"lat": 43.73403,  "lon": -96.62328,  "name": "Sioux Falls, SD",        "climate": "Semi-arid continental"},
    "TBL": {"lat": 40.12498,  "lon": -105.23680, "name": "Table Mountain, CO",     "climate": "Semi-arid, high altitude"},
}

# MATPLOTLIB PALETTE
_P = dict(
    amber  = "#e8b84b",
    blue   = "#5b9bd5",
    text   = "#f0f4f8",
    muted  = "#9db4c8",
    dim    = "#5a7a96",
    bg1    = "#0c1929",
    bg0    = "#07111e",
    border = "#1a2e48",
)

# PAGE CONFIG  (must come before any other st.* call)
st.set_page_config(
    page_title="SOLAD - Solar Irradiance Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Viewport meta - critical for mobile rendering
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">',
    unsafe_allow_html=True,
)

# CSS
# KEY: hide ALL streamlit chrome so position:fixed on .solad-nav works correctly
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
    --bg0:   #07111e;
    --bg1:   #0c1929;
    --bg2:   #111f33;
    --brd:   #1a2e48;
    --brd2:  #243d5c;
    --amber: #e8b84b;
    --amber2:#c9963a;
    --blue:  #5b9bd5;
    --tx:    #f0f4f8;
    --mu:    #9db4c8;
    --dim:   #5a7a96;
    --mono:  'Courier New', Courier, monospace;
    --sans:  'Courier New', Courier, monospace;
    --syne:  'Syne', system-ui, sans-serif;
    --r:     6px;
    --t-xs:  0.6875rem;
    --t-sm:  0.8125rem;
    --t-md:  0.9375rem;
}

/* hide ALL streamlit chrome - required for position:fixed to work */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu { display: none !important; }

/* page */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--bg0) !important;
    color: var(--tx) !important;
    font-family: var(--sans) !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}
section[data-testid="stMain"] > div { background: var(--bg0) !important; }
.block-container { padding-top: 0 !important; }

/* -- NAVBAR -- */
.solad-nav {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: var(--bg1);
    border-bottom: 1px solid var(--brd);
    display: flex;
    align-items: center;
    padding: 0 2.5rem;
    height: 56px;
}
/* brand stays on the left */
.solad-nav .brand {
    font-family: var(--syne);
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--amber);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-decoration: none;
    flex-shrink: 0;
}
/* nav links + meta pushed to the RIGHT */
.solad-nav .nav-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    height: 56px;
}
.solad-nav a.nav-link {
    font-family: var(--mono);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--mu);
    text-decoration: none;
    padding: 0 1.4rem;
    height: 56px;
    display: flex;
    align-items: center;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
}
.solad-nav a.nav-link:hover  { color: var(--tx); border-bottom-color: var(--brd2); }
.solad-nav a.nav-link.active { color: var(--tx); border-bottom-color: var(--amber); }
.solad-nav .nav-meta {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--dim);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-left: 1px solid var(--brd);
    padding-left: 1.4rem;
    margin-left: 0.4rem;
    height: 56px;
}
.solad-nav .nav-dot { width: 7px; height: 7px; background: #4caf7d; border-radius: 50%; }

/* section anchor offset - compensates for fixed navbar */
.section-anchor {
    display: block;
    height: 72px;
    margin-top: -72px;
    visibility: hidden;
}

/* spacer below nav */
.nav-spacer { height: 56px; }

/* force white text on all streamlit elements */
p, div, span, li, td, th,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] div,
[data-testid="stMarkdownContainer"] span { color: var(--tx) !important; }

/* labels */
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    font-family: var(--mono) !important;
    font-size: var(--t-xs) !important;
    font-weight: 500 !important;
    color: var(--mu) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* inputs */
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    background: var(--bg0) !important;
    color: var(--tx) !important;
    border: 1px solid var(--brd) !important;
    border-radius: var(--r) !important;
    font-family: var(--mono) !important;
    font-size: var(--t-sm) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(232,184,75,0.12) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background: var(--bg2) !important; color: var(--mu) !important;
    border-color: var(--brd) !important;
}
[data-testid="stNumberInput"] button:hover { color: var(--amber) !important; }

/* generate button */
.stButton > button {
    background: var(--amber) !important;
    color: #07111e !important;
    border: none !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: var(--t-xs) !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    border-radius: var(--r) !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: var(--amber2) !important; }

/* download button */
.stDownloadButton > button {
    background: transparent !important;
    color: var(--amber) !important;
    border: 1px solid var(--amber) !important;
    font-family: var(--mono) !important;
    font-size: var(--t-xs) !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.4rem 1.2rem !important;
    border-radius: var(--r) !important;
    transition: background 0.15s !important;
}
.stDownloadButton > button:hover { background: rgba(232,184,75,0.10) !important; }

/* expanders */
[data-testid="stExpander"] {
    border: 1px solid var(--brd) !important;
    border-radius: var(--r) !important;
    background: var(--bg1) !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: var(--t-xs) !important;
    font-weight: 500 !important;
    color: var(--mu) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--tx) !important; }
[data-testid="stExpanderDetails"] { padding: 0 1rem 1rem !important; }

/* section rule */
.sec-rule {
    font-family: var(--mono); font-size: var(--t-xs);
    font-weight: 600; color: var(--mu);
    text-transform: uppercase; letter-spacing: 0.10em;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--brd);
    margin: 1.5rem 0 1rem 0;
}

/* Syne section titles & subtitles */
.page-title {
    font-family: var(--syne);
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--tx);
    line-height: 1.2;
    letter-spacing: -0.01em;
    padding-top: 1.5rem;
    padding-bottom: 0.2rem;
}
.page-title span { color: var(--amber); }
.page-subtitle {
    font-family: var(--syne);
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--mu);
    margin-top: 0.35rem;
    letter-spacing: 0.03em;
    padding-bottom: 0.5rem;
}
.section-title {
    font-family: var(--syne);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--tx);
    padding-top: 1.2rem;
    padding-bottom: 0.2rem;
    letter-spacing: -0.01em;
}

/* controls card */
.ctrl-card {
    background: var(--bg1); border: 1px solid var(--brd);
    border-radius: var(--r); padding: 1.2rem 1.4rem; margin-bottom: 1.5rem;
}
.ctrl-card .ctrl-title {
    font-family: var(--mono); font-size: var(--t-xs);
    color: var(--mu); text-transform: uppercase; letter-spacing: 0.11em;
    border-bottom: 1px solid var(--brd); padding-bottom: 0.4rem; margin-bottom: 1rem;
}

/* validation error box */
.val-error {
    background: rgba(220,60,60,0.08);
    border: 1px solid rgba(220,60,60,0.30);
    border-left: 3px solid #dc3c3c;
    border-radius: 0 var(--r) var(--r) 0;
    padding: 0.75rem 1rem;
    font-family: var(--mono);
    font-size: var(--t-xs);
    color: #f08080;
    margin: 0.5rem 0 1rem 0;
}

/* metric cards */
.metric-row { display: flex; gap: 12px; margin: 0.75rem 0 1rem 0; }
.mcard {
    flex: 1; background: var(--bg1); border: 1px solid var(--brd);
    border-top: 2px solid var(--amber); border-radius: var(--r); padding: 1rem 1.1rem;
}
.mcard .lbl { font-family: var(--mono); font-size: var(--t-xs); font-weight: 500;
    color: var(--mu); text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 0.35rem; }
.mcard .val { font-family: var(--mono); font-size: 1.5rem; font-weight: 300;
    color: var(--tx); letter-spacing: -0.02em; line-height: 1.1; }
.mcard .sub { font-family: var(--mono); font-size: var(--t-xs); color: var(--dim); margin-top: 0.2rem; }

/* station table */
.st-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: var(--t-xs); }
.st-table th { color: var(--mu); text-transform: uppercase; letter-spacing: 0.08em;
    font-size: var(--t-xs); font-weight: 500; border-bottom: 1px solid var(--brd);
    padding: 0.22rem 0.6rem; text-align: left; }
.st-table td { color: var(--tx); padding: 0.18rem 0.6rem; border-bottom: 1px solid var(--brd); }
.st-table td:first-child { color: var(--amber); font-weight: 500; }
.st-table tr:last-child td { border-bottom: none; }
.st-table tr:hover td { background: rgba(232,184,75,0.04); }

/* prose */
.prose { font-family: var(--sans); font-size: var(--t-sm); color: var(--tx); line-height: 1.8; }
.prose strong { color: var(--tx); font-weight: 600; }
.prose code { font-family: var(--mono); font-size: 0.75rem; background: var(--bg2);
    border: 1px solid var(--brd); border-radius: 3px; padding: 1px 5px; color: var(--blue); }

/* callout */
.callout { background: rgba(91,155,213,0.07); border: 1px solid rgba(91,155,213,0.20);
    border-left: 2px solid var(--blue); border-radius: 0 var(--r) var(--r) 0;
    padding: 0.75rem 1rem; font-family: var(--sans); font-size: var(--t-sm);
    color: var(--tx); line-height: 1.7; margin: 1rem 0; }

/* step box */
.step-box { background: var(--bg1); border: 1px solid var(--brd);
    border-left: 2px solid var(--amber); border-radius: 0 var(--r) var(--r) 0;
    padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    font-family: var(--sans); font-size: var(--t-sm); color: var(--tx); line-height: 1.8; }
.step-box .step-n { font-family: var(--mono); font-size: var(--t-xs); font-weight: 500;
    color: var(--amber); text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 0.2rem; }
.step-box code { font-family: var(--mono); font-size: 0.75rem; background: var(--bg2);
    border: 1px solid var(--brd); border-radius: 3px; padding: 1px 5px; color: var(--blue); }

/* note box */
.note-box { background: rgba(232,184,75,0.05); border: 1px solid rgba(232,184,75,0.15);
    border-left: 2px solid var(--amber); border-radius: 0 var(--r) var(--r) 0;
    padding: 0.75rem 1rem; font-family: var(--mono); font-size: var(--t-xs);
    color: var(--tx); line-height: 1.8; margin-top: 0.5rem; }

/* empty state */
.empty-state { background: var(--bg1); border: 1px solid var(--brd);
    border-radius: var(--r); padding: 3rem 2rem; text-align: center; margin: 1rem 0; }
.empty-state .title { font-family: var(--syne); font-size: var(--t-md); font-weight: 600;
    color: var(--tx); margin-bottom: 0.4rem; }
.empty-state .sub { font-family: var(--mono); font-size: var(--t-xs); color: var(--mu); }

/* map + result panels */
.panel-flush { background: var(--bg1); border: 1px solid var(--brd); border-radius: var(--r); overflow: hidden; }
.result-panel { background: var(--bg1); border: 1px solid var(--brd); border-radius: var(--r); overflow: hidden; margin-top: 1rem; }
.result-panel-inner { padding: 1rem; }

/* export row */
.export-row { background: var(--bg1); border: 1px solid var(--brd); border-radius: var(--r); padding: 1rem 1.2rem; margin: 1rem 0; }
.export-row .lbl { font-family: var(--mono); font-size: var(--t-xs); font-weight: 500; color: var(--mu); text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 0.25rem; }
.export-row .file-name { font-family: var(--mono); font-size: var(--t-sm); color: var(--tx); font-weight: 500; }
.export-row .file-meta { font-family: var(--mono); font-size: var(--t-xs); color: var(--dim); margin-top: 0.15rem; }

/* loading overlay */
.solad-loader { position: fixed; inset: 0; z-index: 9998; background: rgba(7,17,30,0.92);
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1.5rem; }
.solad-loader .loader-icon { animation: loaderSpin 3s linear infinite; }
@keyframes loaderSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.solad-loader .loader-title { font-family: var(--syne); font-size: 1rem; font-weight: 600; color: var(--tx); letter-spacing: 0.04em; }
.solad-loader .loader-sub { font-family: var(--mono); font-size: var(--t-xs); color: var(--mu); margin-top: -0.75rem; }
.solad-loader .loader-bar-track { width: 220px; height: 2px; background: var(--brd); border-radius: 2px; overflow: hidden; }
.solad-loader .loader-bar-fill { height: 100%; width: 40%; background: var(--amber); border-radius: 2px; animation: loaderSlide 1.6s ease-in-out infinite; }
@keyframes loaderSlide { 0% { transform: translateX(-100%); } 100% { transform: translateX(350%); } }

/* divider */
.hdiv { border: none; border-top: 1px solid var(--brd); margin: 3rem 0 2rem 0; }

/* footer */
.footer { font-family: var(--mono); font-size: var(--t-xs); color: var(--dim);
    text-align: center; padding: 1.5rem 0 1rem; letter-spacing: 0.05em;
    border-top: 1px solid var(--brd); margin-top: 3rem; }

html { scroll-behavior: smooth !important; }

/* title amber spans - more specific than the global span rule */
#solad-title span.amb { color: #e8b84b !important; }

/* About section - stretch columns to equal height */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div:first-child {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* -- MOBILE RESPONSIVENESS -- */
@media (max-width: 768px) {
    /* navbar: wrap to two lines on mobile */
    .solad-nav {
        flex-wrap: wrap !important;
        height: auto !important;
        padding: 0.5rem 1rem !important;
        gap: 0.25rem !important;
    }
    .solad-nav .brand {
        font-size: 0.95rem !important;
        flex-basis: 100% !important;
        padding: 0.2rem 0 !important;
    }
    .solad-nav .nav-right {
        height: auto !important;
        flex-wrap: wrap !important;
        margin-left: 0 !important;
        gap: 0 !important;
        width: 100% !important;
    }
    .solad-nav a.nav-link {
        height: auto !important;
        padding: 0.3rem 0.6rem !important;
        font-size: 0.7rem !important;
        border-bottom: none !important;
        border-left: 2px solid transparent !important;
    }
    .solad-nav a.nav-link.active {
        border-left-color: var(--amber) !important;
        border-bottom: none !important;
    }
    .solad-nav .nav-meta {
        height: auto !important;
        padding-left: 0.6rem !important;
        margin-left: 0 !important;
        font-size: 0.65rem !important;
        border-left: none !important;
        border-top: 1px solid var(--brd) !important;
        width: 100% !important;
        padding-top: 0.3rem !important;
    }

    /* more space below taller navbar */
    .nav-spacer { height: 110px !important; }
    .section-anchor {
        height: 120px !important;
        margin-top: -120px !important;
    }

    /* reduce page horizontal padding */
    [data-testid="stMainBlockContainer"] {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    /* title sizes */
    #solad-title { font-size: 1.35rem !important; }
    .section-title { font-size: 1rem !important; }
    .prose { font-size: 0.8rem !important; line-height: 1.7 !important; }

    /* metric cards: 2-column grid */
    .metric-row {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 8px !important;
    }
    .mcard .val { font-size: 1.1rem !important; }
    .mcard .lbl { font-size: 0.6rem !important; }

    /* export filename wraps cleanly */
    .export-row .file-name {
        word-break: break-all !important;
        font-size: 0.72rem !important;
    }
    .export-row .file-meta { font-size: 0.62rem !important; }

    /* station table: hide climate column on small screens */
    .st-table th:last-child,
    .st-table td:last-child { display: none !important; }

    /* loader bar wider */
    .solad-loader .loader-bar-track { width: 80vw !important; }
    .solad-loader .loader-sub { font-size: 0.6rem !important; text-align: center !important; padding: 0 1rem !important; }

    /* callout & note boxes */
    .callout, .note-box, .step-box {
        font-size: 0.72rem !important;
    }
}

@media (max-width: 480px) {
    /* metric cards: single column on very small phones */
    .metric-row {
        grid-template-columns: 1fr !important;
    }
    #solad-title { font-size: 1.1rem !important; }
    .section-title { font-size: 0.95rem !important; }

    /* reduce divider margins */
    .hdiv { margin: 1.5rem 0 1rem 0 !important; }

    /* empty state */
    .empty-state { padding: 2rem 1rem !important; }
    .empty-state .title { font-size: 0.82rem !important; }
}
</style>
""", unsafe_allow_html=True)


# Model loaders  (local imports deferred inside cached functions)

@st.cache_resource(show_spinner=False)
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner=False)
def load_norm_stats(_cfg):
    from data.physics_utils import IntraDayNormStats, DayFeatureNormStats, ClimateFeatNormStats
    ist = IntraDayNormStats(); ist.load(str(INTRADAY_STATS))
    dfs = DayFeatureNormStats(); dfs.load(str(DAY_FEAT_STATS))
    # Climate stats — optional but needed for correct irradiance quantile scaling.
    # If absent, generate.py falls back to raw features (safe but unscaled).
    cs = None
    if CLIMATE_STATS.exists():
        cs = ClimateFeatNormStats(); cs.load(str(CLIMATE_STATS))
    else:
        log.warning(
            "norm_climate_stats.json not found at %s — "
            "climate features will be unscaled at inference. "
            "Re-run training to generate this file.", CLIMATE_STATS,
        )
    return ist, dfs, cs

@st.cache_resource(show_spinner=False)
def load_models(_cfg):
    from solar_diffusion.vae      import SolarVAE
    from solar_diffusion.denoiser import SolarDenoiser
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = SolarVAE(_cfg).to(device)
    # inference_bundle/ contains clean state_dicts only (no optimizer, no epoch,
    # no ema_shadow) — prepared by prepare_inference_bundle.py which strips all
    # training metadata. weights_only=True is safe and correct here.
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device, weights_only=True))
    vae.eval()
    dn = SolarDenoiser(_cfg).to(device)
    dn.load_state_dict(torch.load(DIFF_PATH, map_location=device, weights_only=True))
    dn.eval()
    return vae, dn, device

@st.cache_resource(show_spinner=False)
def load_schedule_and_transform(_cfg):
    from solar_diffusion.optical_depth import LatentTauTransform, TauNoiseSchedule
    dc = _cfg["diffusion"]
    sch = TauNoiseSchedule(
        num_steps=dc["num_steps"], alpha_bar_min=dc["alpha_bar_min"],
        schedule=dc["noise_schedule"], cosine_s=float(dc.get("cosine_s", 0.008)),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    lt = LatentTauTransform(k_max=_cfg["physics"]["k_max"])
    lt.load(str(TAU_PATH))
    return sch, lt


# INPUT VALIDATION

def validate_inputs(lat, lon, start_date, end_date):
    # 
    # Returns a list of human-readable error strings.
    # Empty list means all inputs are valid.
    # 
    errors = []

    # --- latitude ---
    try:
        lat_f = float(lat)
        if not (LAT_MIN <= lat_f <= LAT_MAX):
            errors.append(
                f"Latitude {lat_f} is out of range. Must be between {LAT_MIN}\u00b0 and {LAT_MAX}\u00b0."
            )
    except (TypeError, ValueError):
        errors.append("Latitude must be a valid number.")

    # --- longitude ---
    try:
        lon_f = float(lon)
        if not (LON_MIN <= lon_f <= LON_MAX):
            errors.append(
                f"Longitude {lon_f} is out of range. Must be between {LON_MIN}\u00b0 and {LON_MAX}\u00b0."
            )
    except (TypeError, ValueError):
        errors.append("Longitude must be a valid number.")

    # --- dates ---
    if not isinstance(start_date, date):
        errors.append("Start date is not a valid date.")
    elif not (DATE_MIN <= start_date <= DATE_MAX):
        errors.append(
            f"Start date {start_date} is out of range. Must be between {DATE_MIN} and {DATE_MAX}."
        )

    if not isinstance(end_date, date):
        errors.append("End date is not a valid date.")
    elif not (DATE_MIN <= end_date <= DATE_MAX):
        errors.append(
            f"End date {end_date} is out of range. Must be between {DATE_MIN} and {DATE_MAX}."
        )

    # --- date ordering & span ---
    if isinstance(start_date, date) and isinstance(end_date, date):
        if end_date < start_date:
            errors.append("End date must be on or after start date.")
        else:
            n = (end_date - start_date).days + 1
            if n > MAX_DAYS:
                errors.append(
                    f"Date range spans {n} days - maximum allowed is {MAX_DAYS} days (1 years). "
                    "Please shorten the range."
                )

    return errors


# Generation helpers
def run_generation(lat, lon, start_date, end_date, cfg,
                   vae, denoiser, schedule, lt, ist, dfs, device):
    from solar_diffusion.generate import generate_sequence
    # Build a runtime config with all required paths resolved to absolute paths.
    # generate_sequence reads climate stats from cfg["paths"]["norm_stats_climate"]
    # internally — no need to pass cs directly.
    # output_csv is set to None so nothing is written to disk during inference.
    cfg_rt = {
        **cfg,
        "paths": {
            **cfg.get("paths", {}),
            "latent_tau_stats":   str(TAU_PATH),
            "regime_gmm":         str(GMM_PATH) if GMM_PATH.exists() else cfg.get("paths", {}).get("regime_gmm", ""),
            "norm_stats_climate": str(CLIMATE_STATS) if CLIMATE_STATS.exists() else "",
            "output_csv":         None,
        },
    }
    gen, cs_ghi, zenith = generate_sequence(
        start_date=str(start_date), end_date=str(end_date),
        lat=float(lat), lon=float(lon),
        vae=vae, denoiser=denoiser, schedule=schedule,
        latent_transform=lt, cfg=cfg_rt,
        intraday_stats=ist, day_feat_stats=dfs, device=device,
    )
    return gen, cs_ghi, zenith


def build_csv(gen, cs_ghi, zenith, dates, cfg):
    # pandas is imported at the top of the file - no deferred import needed
    T = gen.shape[1]
    freq_min = int(24 * 60 / int(cfg["data"].get("inference_steps_per_day", 144)))
    rows = []
    for i, d in enumerate(dates):
        base = pd.Timestamp(str(d))
        for t in range(T):
            k = float(gen[i, t]); g = float(cs_ghi[i, t])
            rows.append({
                "datetime":     (base + pd.Timedelta(minutes=t * freq_min)).strftime("%Y-%m-%d %H:%M"),
                "date":         str(d), "timestep": t,
                "kstar":        round(k, 6),
                "ghi_Wm2":      round(k * g, 4),
                "clearsky_ghi": round(g, 4),
                "zenith_deg":   round(float(zenith[i, t]), 4),
            })
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()


# Matplotlib helpers
def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("none")          # transparent - page bg shows through
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_edgecolor(_P["border"]); sp.set_linewidth(0.9)
    ax.tick_params(colors=_P["text"], labelsize=10.5, length=4, width=0.8)
    ax.grid(True, color=_P["border"], linewidth=0.6, linestyle="-", alpha=0.6, axis="y", zorder=0)
    if title:
        ax.set_title(title, color=_P["text"], fontsize=11,
                     fontfamily="monospace", pad=10, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=_P["muted"], fontsize=10, fontfamily="monospace", labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=_P["muted"], fontsize=10, fontfamily="monospace", labelpad=8)

def make_station_map():
    USA = [
        (-124.7,48.4),(-124.2,46.2),(-124.5,43.0),(-124.1,40.4),(-120.5,34.5),
        (-117.1,32.5),(-114.8,32.5),(-111.0,31.3),(-108.2,31.3),(-106.5,31.8),
        (-104.0,29.6),(-101.0,29.8),(-97.3,25.9),(-96.6,25.8),(-97.1,26.8),
        (-97.4,27.8),(-97.0,28.5),(-94.7,29.4),(-90.0,29.0),(-89.1,30.1),
        (-88.0,30.2),(-85.7,30.1),(-84.9,29.7),(-82.0,29.4),(-81.1,25.1),
        (-80.1,25.1),(-80.1,27.0),(-80.8,28.8),(-80.5,31.0),(-81.4,31.9),
        (-81.2,32.8),(-78.5,33.9),(-75.7,35.2),(-75.5,37.0),(-76.0,37.9),
        (-75.2,38.0),(-74.0,39.5),(-74.2,40.5),(-72.0,41.3),(-71.9,42.0),
        (-70.0,43.0),(-67.0,44.8),(-67.8,47.1),(-69.2,47.5),(-76.9,43.0),
        (-79.0,43.1),(-79.0,42.0),(-82.5,41.8),(-82.7,42.6),(-83.1,42.1),
        (-84.4,46.5),(-87.0,45.1),(-88.0,48.0),(-90.0,48.1),(-92.0,48.5),
        (-95.2,49.0),(-100.0,49.0),(-104.0,49.0),(-110.0,49.0),(-114.1,49.0),
        (-117.0,49.0),(-120.0,49.0),(-123.3,48.6),(-124.7,48.4),
    ]
    STATE_SEGS = [
        [(-104.05,49.0),(-104.05,41.0)],[(-111.05,49.0),(-111.05,41.0)],
        [(-114.05,42.0),(-114.05,34.0)],[(-120.0,42.0),(-120.0,37.0)],
        [(-109.05,45.0),(-109.05,31.3)],[(-100.0,49.0),(-100.0,43.0)],
        [(-96.5,43.5),(-96.5,40.5)],[(-91.0,43.5),(-91.0,40.6)],
        [(-87.5,42.5),(-87.5,39.0)],[(-84.8,39.1),(-84.8,36.6)],
        [(-89.5,35.0),(-89.5,32.0)],[(-94.0,36.5),(-94.0,33.7)],
        [(-103.0,36.5),(-103.0,32.0)],
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_visible(False)
    ax.set_facecolor("none")
    ax.patch.set_visible(False)
    xs = [p[0] for p in USA]; ys = [p[1] for p in USA]
    ax.fill(xs, ys, color=_P["bg1"], zorder=1)
    ax.plot(xs+[xs[0]], ys+[ys[0]], color=_P["border"], lw=0.9, zorder=2)
    for seg in STATE_SEGS:
        ax.plot([p[0] for p in seg],[p[1] for p in seg],
                color=_P["border"], lw=0.5, zorder=2, alpha=0.6)
    LBL_OFF = {"BON":(0.6,0.5),"DRA":(-4.2,0.5),"FPK":(0.6,0.5),
               "GWN":(0.6,-1.1),"PSU":(0.6,0.5),"SXF":(0.6,-1.1)}
    for sid, m in STATION_META.items():
        dx, dy = LBL_OFF.get(sid, (0.6, 0.5))
        ax.plot(m["lon"],m["lat"],"o",ms=8,mfc="none",mec=_P["amber"],mew=1.4,zorder=5)
        ax.plot(m["lon"],m["lat"],"o",ms=3.2,mfc=_P["amber"],mec="none",zorder=6)
        ax.text(m["lon"]+dx,m["lat"]+dy,sid,fontsize=7.5,color=_P["text"],
                fontfamily="monospace",fontweight="bold",ha="left",va="bottom",zorder=7,
                bbox=dict(boxstyle="square,pad=0.20",fc=_P["bg1"],ec=_P["border"],alpha=0.95,lw=0.7))
    ax.set_xlim(-126.0,-66.0); ax.set_ylim(24.0,50.5)
    ax.axis("off"); fig.tight_layout(pad=0.5)
    return fig


def plot_irradiance(gen, cs_ghi, dates, cfg):
    N = len(dates); n = min(N, 3)
    idx = list(range(N)) if N <= 3 else [0, round((N-1)/2), N-1]
    spd = int(cfg["data"].get("inference_steps_per_day", 144))
    hours = np.arange(spd) * (24.0 / spd)
    fig, axes = plt.subplots(1, n, figsize=(7.0*n, 4.8), squeeze=False)
    fig.patch.set_visible(False)
    for pi, di in enumerate(idx):
        ax = axes[0][pi]
        kstar=gen[di]; ghi_cs=cs_ghi[di]; ghi_g=kstar*ghi_cs
        sun=kstar>0; mk=float(kstar[sun].mean()) if sun.any() else 0.0
        peak=float(ghi_cs.max())
        _ax_style(ax)
        ax.fill_between(hours,ghi_cs,alpha=0.10,color=_P["blue"],lw=0)
        ax.plot(hours,ghi_cs,color=_P["blue"],lw=1.8,ls="--",alpha=0.80,label="Clear-sky GHI",zorder=2)
        ax.fill_between(hours,ghi_g,alpha=0.18,color=_P["amber"],lw=0,zorder=3)
        ax.plot(hours,ghi_g,color=_P["amber"],lw=2.4,solid_capstyle="round",label="Generated GHI",zorder=4)
        ax.set_xlim(0,24); ax.set_ylim(-5,peak*1.18 if peak>0 else 120)
        ax.set_xticks([0,6,12,18,24])
        ax.set_xticklabels(["00:00","06:00","12:00","18:00","24:00"],
                           color=_P["text"],fontsize=10.5,fontfamily="monospace")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"{v:.0f}"))
        ax.tick_params(axis="y",colors=_P["text"],labelsize=10.5)
        ax.set_title(str(dates[di]),color=_P["text"],fontsize=12,
                     fontfamily="monospace",pad=10,loc="left",fontweight="bold")
        ax.text(0.98,0.06,f"K\u0305* = {mk:.3f}",transform=ax.transAxes,
                ha="right",va="bottom",color=_P["muted"],fontsize=10.5,fontfamily="monospace")
        if pi==0:
            ax.set_ylabel("GHI (W m\u207b\u00b2)",color=_P["muted"],fontsize=9.5,fontfamily="monospace")
            leg=ax.legend(loc="upper left",frameon=False,fontsize=10.5)
            for t_obj,col in zip(leg.get_texts(),[_P["blue"],_P["amber"]]):
                t_obj.set_color(col); t_obj.set_fontfamily("monospace")
    fig.tight_layout(pad=1.2)
    return fig


def compute_stats(gen, cs_ghi, dates, cfg):
    N,T=gen.shape
    daily_mean=np.array([float(gen[i][gen[i]>0].mean()) if (gen[i]>0).any() else np.nan for i in range(N)])
    kstar_all=gen[gen>0].astype(np.float64)
    max_lag=24; vg_s=np.zeros(max_lag); vg_n=np.zeros(max_lag)
    for i in range(N):
        k=gen[i]
        for h in range(1,max_lag+1):
            a=k[:T-h]; b=k[h:]; mask=(a>0)&(b>0)
            if mask.sum():
                vg_s[h-1]+=0.5*float(np.mean((a[mask]-b[mask])**2)); vg_n[h-1]+=1
    variogram=[float(vg_s[h]/vg_n[h]) if vg_n[h] else None for h in range(max_lag)]
    return dict(daily_mean=daily_mean,kstar_all=kstar_all,variogram=variogram)


def plot_statistics(stats, cfg):
    from scipy.stats import gaussian_kde
    dm=stats["daily_mean"]; ka=stats["kstar_all"]; vg=stats["variogram"]
    kmax=float(cfg["physics"]["k_max"])
    fig,axes=plt.subplots(1,3,figsize=(21,5.0))
    fig.patch.set_visible(False)
    # -- panel 1: daily mean K* --
    ax=axes[0]; valid=~np.isnan(dm); x=np.where(valid)[0]; y=dm[valid]
    ax.fill_between(x,y,alpha=0.15,color=_P["amber"],lw=0)
    ax.plot(x,y,color=_P["amber"],lw=2.0,solid_capstyle="round")
    ax.axhline(0.70,color=_P["dim"],lw=1.0,ls=":",alpha=0.8)
    ax.text(len(dm)*0.01,0.72,"Clear threshold (0.70)",color=_P["dim"],fontsize=10,fontfamily="monospace",va="bottom")
    ax.set_xlim(0,max(len(dm)-1,1)); ax.set_ylim(0,1.35)
    _ax_style(ax,"Daily mean K*  -  sunlit timesteps","Day index","K*")
    # -- panel 2: KDE - Scott bw, 2000-pt grid for a smooth curve --
    ax=axes[1]
    if len(ka)>20:
        kde=gaussian_kde(ka, bw_method="scott")
        xs=np.linspace(0.0, kmax*1.05, 2000); ys=kde(xs)
        ax.fill_between(xs,ys,alpha=0.15,color=_P["amber"],lw=0)
        ax.plot(xs,ys,color=_P["amber"],lw=2.2)
        mk=float(np.mean(ka)); ax.axvline(mk,color=_P["blue"],lw=1.2,ls="--",alpha=0.9)
        ax.text(mk+0.015,float(np.max(ys))*0.90,f"mean = {mk:.3f}",
                color=_P["blue"],fontsize=10,fontfamily="monospace",va="top")
    ax.set_xlim(0,kmax*1.05); ax.set_ylim(bottom=0)
    _ax_style(ax,"K* distribution  -  kernel density estimate","K*  (clear-sky index)","Density")
    # -- panel 3: intraday variogram --
    ax=axes[2]
    lags=[i+1 for i,v in enumerate(vg) if v is not None]
    vals=[v for v in vg if v is not None]
    if lags:
        ax.fill_between(lags,vals,alpha=0.12,color=_P["blue"],lw=0)
        ax.plot(lags,vals,color=_P["blue"],lw=2.0,marker="o",ms=4.5,mfc=_P["blue"],mec="none")
        ax.set_xlim(0.5,max(lags)+0.5); ax.set_ylim(bottom=0)
    _ax_style(ax,"Intraday variogram \u03b3(h)  -  K*  (10-min lags)","Lag  (10-min steps)","\u03b3(h)")
    fig.tight_layout(pad=1.2)
    return fig


# STARTUP: file existence check + model loading
with st.spinner("Loading model weights..."):
    try:
        missing = [p for p in REQUIRED_FILES if not p.exists()]
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(f"Required file(s) not found: {names}")
        cfg                    = load_config()
        ist, dfs, cs           = load_norm_stats(cfg)
        vae, denoiser, dev     = load_models(cfg)
        schedule, lt           = load_schedule_and_transform(cfg)

        # Warn on optional files — non-fatal, generation degrades gracefully.
        _missing_optional = [p for p in OPTIONAL_FILES if not p.exists()]
        if _missing_optional:
            for _p in _missing_optional:
                log.warning(
                    "Optional file not found: %s — location prior will use "
                    "uniform cloudy sub-class split instead of training frequencies.", _p,
                )

        # Warn if climate stats are missing — SiteContextEmbedding and climate
        # features in day_features both require norm_climate_stats.json at inference.
        if cs is None:
            log.warning(
                "norm_climate_stats.json not found — climate features will be "
                "unscaled at inference. Generation will still run but climate "
                "conditioning (Köppen zone + irradiance quantiles) will be degraded."
            )

        # Check bypass_alpha health
        with torch.no_grad():
            for _m in vae.modules():
                if hasattr(_m, "bypass_alpha"):
                    _alpha_val = float(torch.sigmoid(_m.bypass_alpha).item())
                    if _alpha_val > 0.15:
                        log.warning(
                            "bypass_alpha=%.4f > 0.15 — VAE was trained with old 0.0 init. "
                            "Flat-floor artifacts may appear on overcast days. "
                            "Retrain VAE with bypass_alpha init=-3.0 to fix.", _alpha_val,
                        )
                    else:
                        log.info("bypass_alpha=%.4f (healthy, < 0.15).", _alpha_val)

        _model_ok = True
    except Exception as e:
        _model_ok = False
        _model_err = str(e)

if not _model_ok:
    st.error(f"Model initialisation failed - {_model_err}")
    st.stop()


st.markdown(f"""
<nav class="solad-nav">
  <a class="brand" href="#about">SOLAD</a>
  <div class="nav-right">
    <a class="nav-link active" href="#generation" data-section="generation">Generation</a>
    <a class="nav-link" href="#about" data-section="about">About</a>
    <a class="nav-link" href="#data" data-section="data">Instalation</a>
    <div class="nav-meta">
      <div class="nav-dot"></div>
      {APP_VERSION}
    </div>
  </div>
</nav>
<div class="nav-spacer"></div>
""", unsafe_allow_html=True)

components.html("""
<script>
(function() {
  function setActive(id) {
    var links = window.parent.document.querySelectorAll('.nav-link');
    links.forEach(function(a) {
      a.classList.toggle('active', a.dataset.section === id);
    });
  }
  var sections = [''generation', about', 'data'];
  var observer = new window.parent.IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) setActive(entry.target.id);
    });
  }, { rootMargin: '-56px 0px -60% 0px', threshold: 0 });
  function observe() {active
    var found = 0;
    sections.forEach(function(id) {
      var el = window.parent.document.getElementById(id);
      if (el) { observer.observe(el); found++; }
    });
    if (found < sections.length) setTimeout(observe, 300);
  }
  observe();
})();
</script>
""", height=0)


# SECTION 2 - GENERATION
st.markdown('<a class="section-anchor" id="generation"></a>', unsafe_allow_html=True)
st.markdown("""
<a class="section-anchor" id="generation"></a>
<div id='solad-title' style='font-family:Syne,system-ui,sans-serif;font-size:2.1rem;font-weight:700;line-height:1.2;letter-spacing:-0.01em;padding-top:1.5rem;padding-bottom:0.75rem;'>
  <span class='amb'>Sol</span>ar
  <span class='amb'>La</span>tent
  <span class='amb'>D</span>iffusion — Synthetic Irradiance Generator
</div>
""", unsafe_allow_html=True)

# st.markdown("<div class='section-title'>Generation</div>", unsafe_allow_html=True)
# st.markdown("<div class='ctrl-title'>Parameters</div>", unsafe_allow_html=True)
st.caption("The model currently generates data at 10-min resolution")

c_lat, c_lon = st.columns([1, 1], gap="medium")
with c_lat:
    lat = st.number_input("Latitude (\u00b0N)", value=40.05,
                          min_value=-90.0, max_value=90.0, step=0.01, format="%.4f")
with c_lon:
    lon = st.number_input("Longitude (\u00b0E)", value=-88.37,
                          min_value=-180.0, max_value=180.0, step=0.01, format="%.4f")

today = date.today()
c_s, c_e, c_btn = st.columns([1, 1, 0.6], gap="medium")
with c_s:
    start_date = st.date_input("Start date", value=today,
                               min_value=DATE_MIN, max_value=DATE_MAX)
with c_e:
    end_date = st.date_input("End date", value=today + timedelta(days=6),
                             min_value=DATE_MIN, max_value=DATE_MAX)
with c_btn:
    st.markdown("<div style='margin-top:1.55rem;'></div>", unsafe_allow_html=True)
    generate_btn = st.button("Generate", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -- Input validation (runs on every render, before any generation) --
input_errors = validate_inputs(lat, lon, start_date, end_date)

if input_errors:
    for err in input_errors:
        st.markdown(f"<div class='val-error'>&#9888;&nbsp; {err}</div>",
                    unsafe_allow_html=True)
    st.stop()

n_days = (end_date - start_date).days + 1

# Performance warning for long ranges
if n_days > WARN_DAYS:
    st.markdown(
        f"<div class='callout'>Generating <strong>{n_days} days</strong>. "
        "Long sequences may take several minutes on CPU. "
        "Consider splitting into shorter ranges if the session times out.</div>",
        unsafe_allow_html=True,
    )

if generate_btn:
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    loader_slot = st.empty()
    loader_slot.markdown(f"""
    <div class='solad-loader'>
      <svg class='loader-icon' width='56' height='56' viewBox='0 0 56 56' fill='none'>
        <circle cx='28' cy='28' r='10' fill='none' stroke='#e8b84b' stroke-width='1.5' opacity='0.3'/>
        <circle cx='28' cy='28' r='5' fill='#e8b84b' opacity='0.9'/>
        <line x1='28' y1='4'  x2='28' y2='13' stroke='#e8b84b' stroke-width='2' stroke-linecap='round'/>
        <line x1='28' y1='43' x2='28' y2='52' stroke='#e8b84b' stroke-width='2' stroke-linecap='round' opacity='0.4'/>
        <line x1='4'  y1='28' x2='13' y2='28' stroke='#e8b84b' stroke-width='2' stroke-linecap='round' opacity='0.4'/>
        <line x1='43' y1='28' x2='52' y2='28' stroke='#e8b84b' stroke-width='2' stroke-linecap='round' opacity='0.4'/>
        <line x1='9.5' y1='9.5' x2='16' y2='16' stroke='#e8b84b' stroke-width='2' stroke-linecap='round' opacity='0.6'/>
        <line x1='40' y1='40' x2='46.5' y2='46.5' stroke='#e8b84b' stroke-width='2' stroke-linecap='round' opacity='0.2'/>
        <line x1='46.5' y1='9.5' x2='40' y2='16' stroke='#e8b84b' stroke-width='2' stroke-linecap='round' opacity='0.2'/>
        <line x1='16' y1='40' x2='9.5' y2='46.5' stroke='#e8b84b' stroke-width='2' stroke-linecap='round' opacity='0.2'/>
      </svg>
      <div class='loader-title'>Running SOLAD</div>
<div class='loader-sub'>No GPU available - on CPU, generating one full year of data takes approximately 15 minutes</div>
<div class='loader-sub' style='margin-top:0.4rem;color:#e8b84b;letter-spacing:0.06em;'>
  {lat:.4f}&deg;&thinsp;N &nbsp;&middot;&nbsp; {lon:.4f}&deg;&thinsp;E &nbsp;&middot;&nbsp; {start_date} &rarr; {end_date} &nbsp;&middot;&nbsp; {n_days} days
</div>
<div class='loader-bar-track'><div class='loader-bar-fill'></div></div>
    </div>
    """, unsafe_allow_html=True)

    try:
        gen, cs_ghi, zenith = run_generation(
            lat, lon, start_date, end_date,
            cfg, vae, denoiser, schedule, lt, ist, dfs, dev,
        )
        spd       = int(cfg["data"].get("inference_steps_per_day", 144))
        csv_bytes = build_csv(gen, cs_ghi, zenith, dates, cfg)
        stats     = compute_stats(gen, cs_ghi, dates, cfg)
    except Exception as e:
        loader_slot.empty()
        st.error(f"Generation failed - {type(e).__name__}: {e}")
        log.exception("Generation error")
        st.stop()

    loader_slot.empty()

    components.html("""<script>
      setTimeout(function(){
        var t=window.parent?window.parent.document:document;
        var el=t.getElementById('output-anchor');
        if(el){el.scrollIntoView({behavior:'smooth',block:'start'});}
      },400);
    </script>""", height=0)

    st.markdown('<a class="section-anchor" id="output-anchor"></a>', unsafe_allow_html=True)
    st.markdown("<div class='sec-rule'>Output Summary</div>", unsafe_allow_html=True)

    kv=gen[gen>0]; mean_k=float(kv.mean()) if kv.size else 0.0
    std_k=float(kv.std()) if kv.size else 0.0
    peak_ghi=float((gen*cs_ghi).max()) if kv.size else 0.0
    n_rows=n_days*spd

    st.markdown(
        f"<div class='metric-row'>"
        f"<div class='mcard'><div class='lbl'>Days generated</div>"
        f"<div class='val'>{n_days:,}</div><div class='sub'>{start_date} &ndash; {end_date}</div></div>"
        f"<div class='mcard'><div class='lbl'>Mean K* (sunlit)</div>"
        f"<div class='val'>{mean_k:.4f}</div><div class='sub'>clear-sky index</div></div>"
        f"<div class='mcard'><div class='lbl'>Std K* (sunlit)</div>"
        f"<div class='val'>{std_k:.4f}</div><div class='sub'>variability measure</div></div>"
        f"<div class='mcard'><div class='lbl'>Peak GHI</div>"
        f"<div class='val'>{peak_ghi:.0f}</div><div class='sub'>W m&minus;&sup2; full series</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    fname = f"solar_{start_date}_{end_date}_lat{lat:.2f}_lon{lon:.2f}.csv"
    st.markdown(
        f"<div class='export-row'><div class='lbl'>CSV Export</div>"
        f"<div class='file-name'>{fname}</div>"
        f"<div class='file-meta'>{n_rows:,} rows &nbsp;&middot;&nbsp; "
        f"10-min timesteps &nbsp;&middot;&nbsp; kstar, ghi_Wm2, clearsky_ghi, zenith_deg</div></div>",
        unsafe_allow_html=True,
    )
    dl_col, _ = st.columns([1, 2])
    with dl_col:
        st.download_button("Export CSV", csv_bytes, fname, "text/csv", use_container_width=True)

    sample_label = "all days" if n_days <= 3 else "first, middle, and last day"
    st.markdown(
        f"<div class='sec-rule' style='margin-top:1.5rem;'>Generated Days Examples"
        f"&nbsp;&mdash;&nbsp; {sample_label} &nbsp;&middot;&nbsp; ",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='result-panel'><div class='result-panel-inner'>", unsafe_allow_html=True)
    fig_irr = plot_irradiance(gen, cs_ghi, dates, cfg)
    st.pyplot(fig_irr, use_container_width=True); plt.close(fig_irr)
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='sec-rule' style='margin-top:1rem;'>Some Statistics of the Generated  K* Series",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='result-panel'><div class='result-panel-inner'>", unsafe_allow_html=True)
    fig_stat = plot_statistics(stats, cfg)
    st.pyplot(fig_stat, use_container_width=True); plt.close(fig_stat)
    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='empty-state'>
      <div class='title'>Set coordinates and date range above, then Generate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='hdiv'>", unsafe_allow_html=True)


# SECTION 1 - ABOUT
st.markdown('<a class="section-anchor" id="about"></a>', unsafe_allow_html=True)

with st.expander("About", expanded=False):
    # -- Two-column About layout: left = text, right = map (top) + table (bottom) --
    col_text, col_right = st.columns([1.2, 1], gap="medium")
    with col_text:
        st.markdown("""
<div style='
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
    padding-right: 0.5rem;
'>
  <div <br><br> 
  </div>
  <div class='prose' style='font-size:0.88rem;line-height:1.85;font-family:Courier New,Courier,monospace;text-align:justify;'>
  SOLAD Generates 10-mmin realistic synthetic solar irradiance time series at arbitrary locations
  and date ranges -- with no weather data, no sensors, and no observations required
  at inference time. Coordinates and a date range are the only inputs. 
  Currently, it generates the global irradiance, but it can be extended to include both direct and diffuse channels.
  <br><br>
  This implementation is a research prototype and is not intended as a
  production-ready tool. Trained exclusively on 3 years of <strong>SURFRAD</strong> ground measurements from
  seven stations spanning diverse North American climate regimes. No auxiliary
  meteorological variables enter the model at any stage -- only solar irradiance
  records and their derived features. This deliberate minimalism isolates the
  contribution of the architecture and demonstrates that physically realistic
  sequences can be synthesised from solar measurements alone.
  <br><br>
  Generalisation to climatologically distinct or out-of-distribution sites --
  remote archipelagos, tropical monsoon zones -- is not guaranteed;
  retraining or fine-tuning on a modest set of local ground measurements is recommended.
  <br><br>  
  The architecture is designed to be extensible. Conditioning on historical
  observations, sky imagery, or richer inputs requires only a lightweight addition,
  naturally extending the model into a forecaster or a prompt-driven generator -- 
  free-text scene descriptions as generation prompts. 
  Code, weights, and technical details are available in the accompanying <a href='https://github.com/frimane/SOLAD' title='https://github.com/frimane/SOLAD' style='color:#5b9bd5;text-decoration:none;'>GitHub</a>. 
  repository and the <a href='#' title='Link will be available upon publication' style='color:#5b9bd5;text-decoration:none;cursor:help;'>research paper</a>.
  </div>
</div>
""", unsafe_allow_html=True) #<em>The guiding principle: use the minimum to learn and need the minimum to work.</em>
    with col_right:
        # -- Map --
        st.markdown("<div class='sec-rule'>SURFRAD Network</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-flush'>", unsafe_allow_html=True)
        fig_map = make_station_map()
        st.pyplot(fig_map, use_container_width=True)
        plt.close(fig_map)
        st.markdown("</div>", unsafe_allow_html=True)
        # -- Table --
        st.markdown("<div class='sec-rule' style='margin-top:1.2rem;'>Training Stations</div>", unsafe_allow_html=True)
        rows_html = "".join(
            f"<tr><td>{sid}</td><td>{m['name']}</td>"
            f"<td>{m['lat']:.3f}&deg;&thinsp;N</td>"
            f"<td>{m['lon']:.3f}&deg;&thinsp;E</td>"
            f"<td style='color:var(--mu)'>{m['climate']}</td></tr>"
            for sid, m in STATION_META.items()
        )
        st.markdown(
            f"<table class='st-table'>"
            f"<thead><tr><th>ID</th><th>Location</th><th>Lat</th><th>Lon</th><th>Climate</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

st.markdown("<hr class='hdiv'>", unsafe_allow_html=True)
 

# SECTION 3 - INSTALL SOLAD
st.markdown('<a class="section-anchor" id="data"></a>', unsafe_allow_html=True)
st.markdown("<div class='section-title'>Run SOLAD Locally</div>", unsafe_allow_html=True)

st.markdown("""
<div style='color:#9db4c8; font-size:0.92rem; margin-bottom:1.2rem;'>
Follow the steps below to install and run SOLAD on your own machine.
Python&nbsp;3.9 or later is required.
</div>
""", unsafe_allow_html=True)

with st.expander("Step 1 — Clone the repository", expanded=False):
    st.markdown("<div style='color:#9db4c8; margin-bottom:0.5rem;'>Download the source code from GitHub.</div>",
                unsafe_allow_html=True)
    st.code("git clone https://github.com/frimane/SOLAD.git\ncd solad", language="bash")

with st.expander("Step 2 — Create a virtual environment", expanded=False):
    st.markdown("<div style='color:#9db4c8; margin-bottom:0.5rem;'>Isolate dependencies from your system Python.</div>",
                unsafe_allow_html=True)
    st.code("""\
# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\\Scripts\\activate""", language="bash")

with st.expander("Step 3 — Install dependencies", expanded=False):
    st.markdown("<div style='color:#9db4c8; margin-bottom:0.5rem;'>Install all required packages from the provided requirements file.</div>",
                unsafe_allow_html=True)
    st.code("pip install -r requirements.txt", language="bash")

with st.expander("Step 4 — Run the app", expanded=False):
    st.markdown("<div style='color:#9db4c8; margin-bottom:0.5rem;'>Launch the Streamlit interface. The app will open automatically in your browser.</div>",
                unsafe_allow_html=True)
    st.code("streamlit run app.py", language="bash")
 
st.markdown(
    f"<div class='footer'>"
    f"For any questions or feedback, contact me at"
    f"&nbsp;<a href='mailto:Azeddine.frimane@yahoo.com' "
    f"style='color:#5b9bd5;text-decoration:none;'>Azeddine.frimane@yahoo.com</a>"
    f"&nbsp;&middot;&nbsp;"
    f"<span style='color:#e8b84b;font-weight:600;'>SOLAD</span>"
    f"<span style='color:#5a7a96;'>{APP_VERSION}</span>"
    f"</div>",
    unsafe_allow_html=True,
)