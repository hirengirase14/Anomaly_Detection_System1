import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys, os, base64
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.loader   import load_file, detect_label_column, preprocess, auto_detect_sensor_names
from src.pipeline import (
    run_model, evaluate, predict_with_threshold,
    find_best_threshold, auto_contamination,
    UNSUPERVISED_MODELS, SUPERVISED_MODELS
)

# ── Icon loading (PIL optional) ───────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
icon_path  = os.path.join(SCRIPT_DIR, "detection.png")
icon       = "🔮"   # fallback
ICON_HTML  = '<span style="font-size:{FSIZE}">🔮</span>'  # FIX 1: single braces in non-f-string

try:
    from PIL import Image as _PIL_Image
    _img = _PIL_Image.open(icon_path)
    with open(icon_path, "rb") as _f:
        _b64 = base64.b64encode(_f.read()).decode()
    icon      = _img
    # f-string: {{SIZE}} → {SIZE}, {{RADIUS}} → {RADIUS} in the resulting string
    ICON_HTML = f'<img src="data:image/png;base64,{_b64}" style="width:{{SIZE}}px;height:{{SIZE}}px;border-radius:{{RADIUS}};object-fit:cover">'
except Exception:
    pass  # PIL not installed or file missing — use emoji fallback


def icon_html(size=38, radius=11):
    if "{SIZE}" in ICON_HTML:
        return ICON_HTML.replace("{SIZE}", str(size)).replace("{RADIUS}", f"{radius}px")
    return ICON_HTML.replace("{FSIZE}", f"{size*0.6:.0f}px")  # now works correctly


# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Anomaly Detector",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True
dark = st.session_state["dark_mode"]

G1, G2, G3, G4 = "#7c3aed", "#a855f7", "#ec4899", "#f43f5e"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED   = "#f43f5e"
CYAN  = "#06b6d4"

if dark:
    BG       = "#07050f"
    SB_BG    = "#09070e"
    CARD     = "#110e1e"
    CARD2    = "#181430"
    BORDER   = "#21193a"
    BORDER2  = "#382e58"
    TEXT_PRI = "#ede9ff"
    TEXT_SEC = "#8478b0"
    TEXT_MUT = "#3f3660"
    PLOT_BG  = "#110e1e"
    GRID_C   = "#181430"
    SHADOW   = "0 8px 32px rgba(0,0,0,0.6)"
    GLASS_BG = "rgba(255,255,255,0.03)"
    GLASS_BD = "rgba(255,255,255,0.07)"
else:
    BG       = "#f6f4ff"
    SB_BG    = "#efecff"
    CARD     = "#ffffff"
    CARD2    = "#f0edff"
    BORDER   = "#ddd6ff"
    BORDER2  = "#b8abf0"
    TEXT_PRI = "#17103a"
    TEXT_SEC = "#4e4280"
    TEXT_MUT = "#9e96c8"
    PLOT_BG  = "#ffffff"
    GRID_C   = "#ede8ff"
    SHADOW   = "0 4px 24px rgba(80,60,160,0.10)"
    GLASS_BG = "rgba(255,255,255,0.75)"
    GLASS_BD = "rgba(160,140,240,0.35)"

HEADER_C         = TEXT_PRI
ACTIVE_THEME_LBL = "Dark" if dark else "Light"

# FIX 6: removed st._config.set_option (private API — unreliable)

# ══════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Fira+Code:wght@400;500;600&display=swap');
@keyframes fadeUp   {{ from{{opacity:0;transform:translateY(16px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes fadeLeft {{ from{{opacity:0;transform:translateX(-12px)}} to{{opacity:1;transform:translateX(0)}} }}
@keyframes gradFlow {{ 0%,100%{{background-position:0% 50%}} 50%{{background-position:100% 50%}} }}
@keyframes glow     {{ 0%,100%{{box-shadow:0 0 0 0 {G2}44}} 50%{{box-shadow:0 0 16px 4px {G2}33}} }}
@keyframes countUp  {{ from{{opacity:0;transform:scale(0.85)}} to{{opacity:1;transform:scale(1)}} }}
html, body, [class*="css"],
[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],[data-testid="stMain"],
[data-testid="stHeader"],[data-testid="stToolbar"],
.stApp,.stMain {{
  font-family:'Outfit',sans-serif !important;
  background:{BG} !important;
  color:{TEXT_PRI} !important;
}}
[data-testid="stForm"],[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],section[data-testid="stSidebar"] {{
  background:transparent !important;
}}
.main .block-container {{
  padding:0.5rem 2rem 3rem 2rem !important;
  max-width:1440px !important;
  animation:fadeUp .45s ease both;
}}
[data-testid="stSidebar"] {{
  background:{SB_BG} !important;
  border-right:1px solid {BORDER} !important;
  width:270px !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding:0 !important; overflow-x:hidden !important; }}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p {{ color:{TEXT_SEC} !important; font-size:0.8rem !important; }}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p {{ color:{TEXT_PRI} !important; font-size:0.8rem !important; -webkit-text-fill-color:{TEXT_PRI} !important; }}
/* Sidebar native widget text */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{
  color:{TEXT_PRI} !important; -webkit-text-fill-color:{TEXT_PRI} !important; }}
div[data-testid="stFileUploader"] {{
  border:1.5px dashed {G1}55 !important; border-radius:12px !important;
  background:{CARD} !important;
}}
div[data-testid="stFileUploader"]:hover {{ border-color:{G2} !important; }}
/* ── Theme toggle buttons (small icon buttons in sidebar header) ── */
[data-testid="stSidebar"] .stButton button {{
  background:{CARD} !important;
  border:1px solid {BORDER2} !important;
  border-radius:8px !important;
  padding:4px 8px !important;
  font-size:1rem !important;
  font-weight:400 !important;
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
  box-shadow:none !important;
  width:auto !important;
  min-width:32px !important;
  animation:none !important;
}}
[data-testid="stSidebar"] .stButton button:hover {{
  border-color:{G2} !important;
  background:{CARD2} !important;
}}
[data-testid="stSidebar"] .stButton {{
  width:auto !important;
  padding:0 !important;
}}
/* ── Run button in main area (full gradient) ── */
.run-btn-wrap .stButton button {{
  background:linear-gradient(135deg,{G1},{G2},{G3},{G4}) !important;
  background-size:300% 300% !important;
  animation:gradFlow 5s ease infinite !important;
  border:none !important;
  border-radius:14px !important;
  font-size:1.05rem !important;
  font-weight:800 !important;
  padding:18px 0 !important;
  box-shadow:0 6px 28px {G2}66 !important;
  width:100% !important;
  letter-spacing:.3px !important;
}}
.run-btn-wrap .stButton button p,
.run-btn-wrap .stButton button span {{
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  background:transparent !important;
  font-size:inherit !important;
  font-weight:inherit !important;
  padding:0 !important;
  width:auto !important;
  display:inline !important;
}}
.run-btn-wrap .stButton button:hover {{
  box-shadow:0 8px 36px {G2}88 !important;
  transform:translateY(-1px) !important;
}}
[data-testid="stDownloadButton"] button {{
  background:{CARD} !important; border:1px solid {BORDER2} !important;
  color:{G2} !important; border-radius:8px !important; font-size:.82rem !important;
  padding:8px 18px !important; width:auto !important;
}}
.stSelectbox > div > div {{
  background:{CARD} !important; border:1px solid {BORDER} !important;
  border-radius:10px !important; color:{TEXT_PRI} !important;
}}
[data-testid="stDataFrame"] {{ border-radius:14px !important; overflow:hidden !important; border:1px solid {BORDER} !important; }}
[data-testid="stProgress"] > div > div {{ background:linear-gradient(90deg,{G1},{G2},{G3}) !important; }}
hr {{ border-color:{BORDER} !important; margin:.6rem 0 !important; }}
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER2}; border-radius:4px; }}
::-webkit-scrollbar-thumb:hover {{ background:{G2}; }}
[data-testid="stTextInput"] input {{
  background:{CARD} !important; border:1px solid {BORDER} !important;
  color:{TEXT_PRI} !important; border-radius:8px !important;
}}
/* ══ CHECKBOX ══════════════════════════════════════════ */
/* Label text — color only, no width/height/sizing */
[data-testid="stCheckbox"] label p {{
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
}}
/* Box indicator: only target the first-child div (visual box), not the label span */
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child {{
  background-color:{CARD} !important;
  border:1.5px solid {BORDER2} !important;
  border-radius:4px !important;
  flex-shrink:0 !important;
  transition:border-color .15s ease, background-color .15s ease !important;
}}
/* Hover: border turns accent */
[data-testid="stCheckbox"]:hover [data-baseweb="checkbox"] > div:first-child {{
  border-color:{G2} !important;
}}
/* Checked: fill with accent */
[data-testid="stCheckbox"] [data-checked="true"] > div:first-child {{
  background-color:{G2} !important;
  border-color:{G2} !important;
}}
/* Remove Streamlit default blue focus ring */
[data-testid="stCheckbox"] [data-baseweb="checkbox"]:focus-within > div:first-child {{
  box-shadow:none !important;
  outline:none !important;
}}

/* ══ SLIDER ═══════════════════════════════════════════ */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p {{
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
}}
[data-testid="stSlider"] [data-testid="stTickBar"] span {{
  color:{TEXT_SEC} !important;
  -webkit-text-fill-color:{TEXT_SEC} !important;
}}
[data-testid="stSlider"] [role="slider"] {{
  background:{G2} !important;
  border-color:{G2} !important;
}}

/* ══ TEXT INPUT ════════════════════════════════════════ */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label p {{
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
}}

/* ══ FILE UPLOADER ═════════════════════════════════════ */
[data-testid="stFileUploader"] label p {{
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
}}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {{
  color:{TEXT_SEC} !important;
  -webkit-text-fill-color:{TEXT_SEC} !important;
}}
[data-testid="stFileUploaderDropzone"] {{
  background:{CARD} !important;
  border:1.5px dashed {G1}77 !important;
  border-radius:12px !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
  background:{G1}18 !important;
  border:1px solid {G1}55 !important;
  color:{G1} !important;
  -webkit-text-fill-color:{G1} !important;
  border-radius:8px !important;
}}

/* ══ SELECTBOX ════════════════════════════════════════ */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p {{
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] div {{
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
  background:{CARD} !important;
}}

/* ══ DATAFRAME / TABLE ═══════════════════════════════ */
[data-testid="stDataFrame"] {{
  border-radius:14px !important;
  overflow:hidden !important;
  border:1px solid {BORDER} !important;
}}
/* DataFrame header row */
[data-testid="stDataFrame"] thead tr th,
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] .header-cell {{
  background:{CARD2} !important;
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
  font-weight:600 !important;
  border-bottom:1px solid {BORDER2} !important;
}}
/* DataFrame body cells */
[data-testid="stDataFrame"] tbody tr td,
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] .cell {{
  background:{CARD} !important;
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
  border-color:{BORDER} !important;
}}
/* DataFrame alternating row */
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {{
  background:{CARD2} !important;
}}
/* DataFrame canvas — Streamlit uses canvas for its table */
[data-testid="stDataFrame"] canvas {{
  border-radius:12px !important;
}}
/* Glide data editor full override */
.dvn-scroller,
[class*="dvn-"] {{
  background:{CARD} !important;
  color:{TEXT_PRI} !important;
}}

/* ══ ALERT / BANNER BOXES ════════════════════════════ */
[data-testid="stAlert"] {{
  background:{CARD} !important;
  border:1px solid {BORDER} !important;
  color:{TEXT_PRI} !important;
}}
[data-testid="stAlert"] p {{
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
}}

/* ══ CAPTIONS ════════════════════════════════════════ */
[data-testid="stCaptionContainer"] p,
.stCaption p {{
  color:{TEXT_MUT} !important;
  -webkit-text-fill-color:{TEXT_MUT} !important;
}}

/* ══ LABELS (scoped — avoid breaking gradient text) ══ */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {{
  color:{TEXT_PRI} !important;
  -webkit-text-fill-color:{TEXT_PRI} !important;
}}

/* ══ METRIC/EVAL CARD VALUES — keep dynamic color ════ */
.metric-val {{ -webkit-text-fill-color:inherit !important; }}
.eval-val   {{ -webkit-text-fill-color:inherit !important; }}
.cv         {{ -webkit-text-fill-color:{G2} !important; }}
/* ── components ── */
.sb-brand {{ display:flex; align-items:center; gap:11px; padding:16px 14px 12px; }}
.sb-name  {{ font-size:1.05rem; font-weight:800; letter-spacing:-.3px; background:linear-gradient(135deg,{G1},{G2},{G3}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }}
.sb-tag   {{ font-size:.66rem; color:{TEXT_MUT}; margin-top:1px; -webkit-text-fill-color:{TEXT_MUT}; font-weight:500; }}
.sb-label {{ font-family:'Fira Code',monospace; font-size:.62rem; font-weight:600; text-transform:uppercase; letter-spacing:2.5px; background:linear-gradient(90deg,{G1},{G3}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:10px; display:block; }}
.sb-info  {{ background:{GLASS_BG}; border:1px solid {GLASS_BD}; border-radius:8px; padding:8px 11px; font-size:.75rem; color:{TEXT_SEC}; line-height:1.5; margin-bottom:8px; }}
.sb-model-group {{ font-size:.72rem; font-weight:600; color:{TEXT_SEC}; text-transform:uppercase; letter-spacing:1px; margin:8px 0 4px; }}
.page-hdr {{ display:flex; align-items:center; justify-content:space-between; padding-bottom:1.2rem; border-bottom:1px solid {BORDER}; animation:fadeUp .4s ease both; }}
.ph-left  {{ display:flex; align-items:center; gap:15px; }}
.ph-logo  {{ width:52px; height:52px; border-radius:15px; flex-shrink:0; overflow:hidden; box-shadow:0 6px 28px {G2}55; animation:glow 3s ease infinite; }}
.ph-title {{ font-size:1.9rem; font-weight:900; letter-spacing:-1px; line-height:1; background:linear-gradient(135deg,{G1} 0%,{G2} 45%,{G3} 80%,{G4} 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }}
.ph-sub   {{ font-size:.76rem; color:{TEXT_MUT}; margin-top:4px; }}
.ph-right {{ display:flex; align-items:center; gap:8px; }}
.ph-chip  {{ padding:4px 12px; border-radius:20px; font-size:.67rem; font-weight:700; text-transform:uppercase; letter-spacing:.5px; background:{G1}1a; border:1px solid {G1}44; color:{G2}; }}
.sec-hdr  {{ display:flex; align-items:center; gap:9px; font-family:'Fira Code',monospace; font-size:.64rem; font-weight:600; text-transform:uppercase; letter-spacing:3px; color:{TEXT_SEC}; margin:1.8rem 0 .9rem; padding-bottom:9px; border-bottom:1px solid {BORDER}; }}
.sec-bar  {{ width:3px; height:14px; border-radius:2px; flex-shrink:0; background:linear-gradient(180deg,{G1},{G3}); }}
.step-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:20px; padding:32px 20px; text-align:center; position:relative; overflow:hidden; transition:all .3s cubic-bezier(.34,1.56,.64,1); animation:fadeUp .5s ease both; box-shadow:{SHADOW}; }}
.step-card:hover {{ transform:translateY(-6px); border-color:{G2}66; box-shadow:0 20px 50px {G2}25; }}
.step-ico   {{ font-size:2.4rem; margin-bottom:14px; display:block; }}
.step-num   {{ font-family:'Fira Code',monospace; font-size:.62rem; font-weight:600; text-transform:uppercase; letter-spacing:2px; margin-bottom:7px; background:linear-gradient(90deg,{G1},{G3}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }}
.step-title {{ font-size:.97rem; font-weight:700; color:{TEXT_PRI}; margin-bottom:7px; }}
.step-desc  {{ font-size:.77rem; color:{TEXT_SEC}; line-height:1.55; }}
.feat-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:14px; padding:20px 14px; text-align:center; transition:all .25s ease; animation:fadeUp .5s ease both; box-shadow:{SHADOW}; height:180px; display:flex; flex-direction:column; align-items:center; justify-content:center; box-sizing:border-box; }}
.feat-card:hover {{ transform:translateY(-3px); border-color:{G2}55; }}
.feat-ico {{ font-size:1.75rem; margin-bottom:9px; }} .feat-name {{ font-size:.86rem; font-weight:700; color:{TEXT_PRI}; margin-bottom:5px; }} .feat-desc {{ font-size:.73rem; color:{TEXT_SEC}; line-height:1.45; }}
.file-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:13px 14px; display:flex; align-items:center; gap:11px; transition:all .2s ease; animation:fadeLeft .4s ease both; box-shadow:{SHADOW}; }}
.file-card:hover {{ border-color:{G2}55; }}
.file-ico  {{ width:36px; height:36px; border-radius:9px; background:{G1}1a; border:1px solid {G1}33; display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }}
.file-name {{ font-family:'Fira Code',monospace; font-size:.76rem; color:{G2}; word-break:break-all; }}
.file-meta {{ font-size:.68rem; color:{TEXT_MUT}; margin-top:2px; }}
.metric-card {{ background:{GLASS_BG}; backdrop-filter:blur(12px); border:1px solid {GLASS_BD}; border-radius:18px; padding:22px 16px; text-align:center; position:relative; overflow:hidden; transition:all .3s cubic-bezier(.34,1.56,.64,1); animation:fadeUp .45s ease both; box-shadow:{SHADOW}; }}
.metric-card:hover {{ transform:translateY(-4px); box-shadow:0 16px 40px {G2}28; }}
.metric-glow {{ position:absolute; top:-10px; right:-10px; width:80px; height:80px; border-radius:50%; opacity:.12; filter:blur(20px); pointer-events:none; }}
.metric-ico {{ font-size:1.15rem; margin-bottom:8px; opacity:.75; }}
.metric-val {{ font-family:'Fira Code',monospace; font-size:1.9rem; font-weight:700; line-height:1; margin-bottom:6px; animation:countUp .5s ease both; }}
.metric-lbl {{ font-size:.67rem; color:{TEXT_MUT}; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; }}
.eval-card  {{ background:{GLASS_BG}; backdrop-filter:blur(10px); border:1px solid {GLASS_BD}; border-radius:14px; padding:16px 12px; text-align:center; transition:all .2s ease; animation:fadeUp .4s ease both; }}
.eval-card:hover {{ border-color:{BORDER2}; transform:translateY(-2px); }}
.eval-val {{ font-family:'Fira Code',monospace; font-size:1.35rem; font-weight:600; line-height:1; }}
.eval-lbl {{ font-size:.66rem; color:{TEXT_MUT}; text-transform:uppercase; letter-spacing:1px; margin-top:5px; }}
.badge    {{ display:inline-block; padding:2px 9px; border-radius:20px; font-size:.62rem; font-weight:700; letter-spacing:.4px; text-transform:uppercase; }}
.b-grad   {{ background:{G1}25; color:{G2}; border:1px solid {G1}44; }}
.b-green  {{ background:{GREEN}1a; color:{GREEN}; border:1px solid {GREEN}44; }}
.b-red    {{ background:{RED}1a; color:{RED}; border:1px solid {RED}44; }}
.b-amber  {{ background:{AMBER}1a; color:{AMBER}; border:1px solid {AMBER}44; }}
.b-cyan   {{ background:{CYAN}1a; color:{CYAN}; border:1px solid {CYAN}44; }}
.info-banner    {{ background:{GLASS_BG}; border:1px solid {GLASS_BD}; border-left:2px solid {G1}; border-radius:0 10px 10px 0; padding:10px 14px; font-size:.81rem; color:{TEXT_SEC}; margin:7px 0; line-height:1.5; }}
.success-banner {{ background:{GREEN}0d; border:1px solid {GREEN}33; border-radius:11px; padding:11px 15px; font-size:.81rem; color:{GREEN}; margin:9px 0; line-height:1.5; }}
.tune-banner    {{ background:{GLASS_BG}; border:1px solid {BORDER2}; border-radius:12px; padding:12px 16px; margin:9px 0; font-size:.83rem; color:{TEXT_SEC}; display:flex; align-items:center; flex-wrap:wrap; gap:8px; }}
.cont-banner    {{ background:{GLASS_BG}; border:1px solid {BORDER}; border-radius:11px; padding:10px 15px; font-size:.81rem; color:{TEXT_SEC}; margin-bottom:1rem; display:flex; align-items:center; gap:7px; }}
.cv {{ font-family:'Fira Code',monospace; color:{G2}; font-weight:600; }}
.chart-glass {{ background:{GLASS_BG}; backdrop-filter:blur(12px); border:1px solid {GLASS_BD}; border-radius:18px; padding:4px; box-shadow:{SHADOW}; overflow:hidden; animation:fadeUp .5s ease both; }}

.cfg-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:18px; padding:22px 20px 18px; box-shadow:{SHADOW}; margin-bottom:0; }}
.cfg-card-title {{ display:flex; align-items:center; gap:9px; font-size:.82rem; font-weight:700; color:{TEXT_PRI}; margin-bottom:16px; padding-bottom:10px; border-bottom:1px solid {BORDER}; }}
.cfg-ico {{ font-size:1.1rem; }}
.upload-zone {{ background:{CARD}; border:2px dashed {G1}55; border-radius:18px; padding:28px 20px; text-align:center; box-shadow:{SHADOW}; transition:border-color .2s; }}
.upload-zone:hover {{ border-color:{G2}; }}
.upload-title {{ font-size:1rem; font-weight:700; color:{TEXT_PRI}; margin-bottom:6px; }}
.upload-sub {{ font-size:.76rem; color:{TEXT_SEC}; line-height:1.6; }}
.file-pill {{ display:inline-flex; align-items:center; gap:7px; background:{G1}18; border:1px solid {G1}33; border-radius:30px; padding:5px 13px 5px 8px; margin:4px; font-size:.74rem; color:{G2}; font-family:'Fira Code',monospace; }}
.run-btn-wrap button {{ background:linear-gradient(135deg,{G1},{G2},{G3},{G4}) !important; background-size:300% 300% !important; animation:gradFlow 5s ease infinite !important; border:none !important; border-radius:14px !important; font-size:1rem !important; font-weight:700 !important; padding:18px 0 !important; box-shadow:0 6px 28px {G2}55 !important; width:100% !important; color:#fff !important; -webkit-text-fill-color:#fff !important; }}
.step-pill {{ display:flex; align-items:flex-start; gap:13px; padding:14px 16px; background:{GLASS_BG}; border:1px solid {GLASS_BD}; border-radius:12px; margin-bottom:10px; }}
.step-pill-num {{ width:28px; height:28px; border-radius:50%; background:linear-gradient(135deg,{G1},{G3}); display:flex; align-items:center; justify-content:center; font-size:.7rem; font-weight:800; color:#fff; flex-shrink:0; margin-top:1px; }}
.step-pill-body {{ flex:1; }}
.step-pill-title {{ font-size:.82rem; font-weight:700; color:{TEXT_PRI}; margin-bottom:3px; }}
.step-pill-desc {{ font-size:.72rem; color:{TEXT_SEC}; line-height:1.5; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════
st.markdown(f"""
<div class="page-hdr">
  <div class="ph-left">
    <div class="ph-logo">{icon_html(52, 15)}</div>
    <div>
      <div class="ph-title">Anomaly Detector</div>
      <div class="ph-sub">ML-Powered Industrial Fault Detection &nbsp;·&nbsp; Upload · Configure · Detect · Analyze</div>
    </div>
  </div>
  <div class="ph-right">
    <span class="ph-chip">v2.0</span>
    <span class="ph-chip">ML Platform</span>
  </div>
</div>
<div style="border-bottom:1px solid {BORDER};margin:.2rem 0 1.2rem"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  SIDEBAR — Steps guide only
# ══════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand + theme toggle row ──────────────────────
    st.markdown(f'''
    <div class="sb-brand">
      {icon_html(38, 11)}
      <div style="flex:1">
        <div class="sb-name">Anomaly Detector</div>
        <div class="sb-tag">ML Detection Platform</div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # Theme toggle buttons — side by side, visible in both modes
    t1, t2 = st.columns(2)
    with t1:
        if st.button(
            "🌙  Dark" if not dark else "🌙  Dark  ✓",
            key="btn_dark",
            help="Switch to dark mode",
            use_container_width=True
        ):
            st.session_state["dark_mode"] = True
            st.rerun()
    with t2:
        if st.button(
            "☀️  Light  ✓" if not dark else "☀️  Light",
            key="btn_light",
            help="Switch to light mode",
            use_container_width=True
        ):
            st.session_state["dark_mode"] = False
            st.rerun()

    st.markdown(f'<div style="border-bottom:1px solid {BORDER};margin:12px 0 18px"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="padding:0 16px"><span class="sb-label">How to use</span></div>', unsafe_allow_html=True)

    steps = [
        ("1", "Upload Dataset",
         "Click 'Browse files' or drag & drop your data files (CSV, NPY, Excel, JSON…). For MSL/SMAP upload all 3 files together."),
        ("2", "Configure Settings",
         "Set anomaly rate, threshold mode, and toggle time series mode in the Core Settings card."),
        ("3", "Select Models",
         "Pick one or more models from the Model Selection card. Isolation Forest is recommended for unlabelled data."),
        ("4", "Run Detection",
         "Click the Run Detection button. Results, charts, and the prediction table will appear below."),
        ("5", "Export Results",
         "Download predictions as CSV using the Download button at the bottom of the results."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-pill">
          <div class="step-pill-num">{num}</div>
          <div class="step-pill-body">
            <div class="step-pill-title">{title}</div>
            <div class="step-pill-desc">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin:18px 16px 0;padding:12px 14px;background:{GLASS_BG};border:1px solid {GLASS_BD};border-radius:10px;font-size:.72rem;color:{TEXT_SEC};line-height:1.6">
    <strong style="color:{G2}">Supported formats</strong><br>
    CSV · Excel · JSON · NPY · NPZ<br>Parquet · TSV · TXT · HDF5
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def identify_roles(files):
    roles = {"train": None, "test": None, "label": None, "single": None}
    if len(files) == 1:
        roles["single"] = files[0]
        return roles
    for f in files:
        n = f.name.lower()
        if   "train" in n: roles["train"] = f
        elif "label" in n: roles["label"] = f
        elif "test"  in n: roles["test"]  = f
    if not roles["train"] and not roles["test"]:
        roles["single"] = files[0]
    return roles

GRAD_CMAP = LinearSegmentedColormap.from_list("iq", [G1, G2, G3, G4])

def sfig(fig, axes=None):
    fig.patch.set_facecolor(PLOT_BG)
    for ax in (axes or fig.get_axes()):
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        ax.xaxis.label.set_color(TEXT_SEC)
        ax.yaxis.label.set_color(TEXT_SEC)
        ax.title.set_color(HEADER_C)
        for s in ax.spines.values():
            s.set_edgecolor(BORDER)
    return fig

def glass_chart(fig):
    st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════
#  ROW 1 — Upload Card (full width)
# ══════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Step 1 — Upload Dataset</div>', unsafe_allow_html=True)

with st.container():
    st.markdown(f'<div class="cfg-card"><div class="cfg-card-title"><span class="cfg-ico">📂</span>Dataset Upload</div></div>', unsafe_allow_html=True)
    up_col1, up_col2 = st.columns([3, 2])
    with up_col1:
        uploaded_files = st.file_uploader(
            "Drop your data files here",
            type=["csv","xlsx","xls","json","npy","npz","parquet","tsv","txt","h5","hdf5"],
            accept_multiple_files=True,
            help="Upload all files together for multi-file datasets (e.g. MSL train + test + label)"
        )
    with up_col2:
        if uploaded_files:
            st.markdown(f'<div style="padding-top:8px">', unsafe_allow_html=True)
            for f in uploaded_files:
                ext  = f.name.rsplit(".", 1)[-1].upper() if "." in f.name else "FILE"
                size = f"{round(f.size/1024,1)} KB" if f.size < 1024*1024 else f"{round(f.size/1024/1024,1)} MB"
                st.markdown(f'''
                <div class="file-card" style="margin-bottom:8px">
                  <div class="file-ico">📄</div>
                  <div>
                    <div class="file-name">{f.name}</div>
                    <div class="file-meta">{size} &nbsp;·&nbsp; <span class="badge b-grad">{ext}</span></div>
                  </div>
                </div>''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding:20px 10px;text-align:center;color:{TEXT_MUT};font-size:.82rem;line-height:1.8">
              📁 No files uploaded yet<br>
              <span style="font-size:.72rem">Supports CSV · Excel · JSON · NPY · NPZ · Parquet · HDF5</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  ROW 2 — Three Config Cards side by side
# ══════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Step 2 — Configure</div>', unsafe_allow_html=True)

col_core, col_thresh, col_models = st.columns(3)

# ── Card A: Core Settings ─────────────────────────────
with col_core:
    st.markdown(f'<div class="cfg-card"><div class="cfg-card-title"><span class="cfg-ico">⚙️</span>Core Settings</div></div>', unsafe_allow_html=True)
    contamination_input = st.slider("Anomaly rate (manual)", 0.01, 0.5, 0.1, 0.01,
                                    help="Fraction of data expected to be anomalous")
    auto_cont      = st.checkbox("Auto-detect contamination from labels", value=True)
    is_timeseries  = st.checkbox("Time series data (skip deduplication)", value=False)
    label_col_hint = st.text_input("Label column name (optional)",
                                   placeholder="e.g. faulty, label, anomaly")
    export_scores  = st.checkbox("Include anomaly scores in CSV export", value=True)

# ── Card B: Threshold ─────────────────────────────────
with col_thresh:
    st.markdown(f'<div class="cfg-card"><div class="cfg-card-title"><span class="cfg-ico">🎯</span>Threshold</div></div>', unsafe_allow_html=True)
    use_auto_threshold   = st.checkbox("Auto-find best threshold (max F1)", value=True,
                                       help="Scans 1–50% and picks the threshold with highest F1 score")
    use_manual_threshold = st.checkbox("Manual threshold override", value=False)
    manual_threshold     = st.slider("Manual threshold (%)", 1, 50, 10, 1,
                                     disabled=not use_manual_threshold,
                                     help="Top X% of anomaly scores flagged as anomalies")
    st.markdown(f"""
    <div style="margin-top:12px;padding:10px 12px;background:{GLASS_BG};border:1px solid {GLASS_BD};border-radius:10px;font-size:.74rem;color:{TEXT_SEC};line-height:1.6">
    💡 <strong>Tip:</strong> Use Auto threshold for best accuracy. Manual is useful when you know your expected anomaly rate.
    </div>""", unsafe_allow_html=True)

# ── Card C: Model Selection ───────────────────────────
with col_models:
    st.markdown(f'<div class="cfg-card"><div class="cfg-card-title"><span class="cfg-ico">🤖</span>Model Selection</div></div>', unsafe_allow_html=True)
    auto_tune_if = st.checkbox("Auto-tune Isolation Forest", value=True,
                               help="Tests 7 hyperparameter configs and picks the best one")
    st.markdown(f'<div style="font-size:.72rem;font-weight:600;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px">Unsupervised (no labels needed)</div>', unsafe_allow_html=True)
    selected_unsupervised = []
    for name, desc in UNSUPERVISED_MODELS.items():
        if st.checkbox(name, value=(name == "Isolation Forest"), key=f"un_{name}", help=desc):
            selected_unsupervised.append(name)
    st.markdown(f'<div style="font-size:.72rem;font-weight:600;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px">Supervised <span style="color:{TEXT_MUT};font-weight:400;text-transform:none">(needs labels)</span></div>', unsafe_allow_html=True)
    selected_supervised = []
    for name, desc in SUPERVISED_MODELS.items():
        if st.checkbox(name, value=False, key=f"su_{name}", help=desc):
            selected_supervised.append(name)


# ══════════════════════════════════════════════════════
#  ROW 3 — Run Button Card (full width, centered)
# ══════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Step 3 — Run</div>', unsafe_allow_html=True)

run_l, run_m, run_r = st.columns([1, 2, 1])
with run_m:
    st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)
    run_btn = st.button("🚀  Run Detection", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not uploaded_files and not st.session_state.get("ready"):
    st.markdown(f"""
    <div style="margin-top:2rem;padding:28px;background:{GLASS_BG};border:1px solid {GLASS_BD};border-radius:18px;text-align:center">
      <div style="font-size:2rem;margin-bottom:10px">📊</div>
      <div style="font-size:1rem;font-weight:700;color:{TEXT_PRI};margin-bottom:6px">No data uploaded yet</div>
      <div style="font-size:.8rem;color:{TEXT_SEC}">Upload your dataset above and click Run Detection to see anomaly results, charts, and model evaluation here.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════
if run_btn:
    selected_models = selected_unsupervised + selected_supervised

    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    with st.spinner("Loading and processing files…"):
        try:
            roles = identify_roles(uploaded_files)

            # ===============================
            # CASE 1: Train + Test files
            # ===============================
            if roles["train"] and roles["test"]:

                train_raw = load_file(roles["train"], roles["train"].name)
                test_raw  = load_file(roles["test"],  roles["test"].name)

                # 🔥 AUTO SENSOR NAMING — pass filename for dataset-specific detection
                new_cols = auto_detect_sensor_names(train_raw, roles["train"].name)
                train_raw.columns = new_cols
                test_raw.columns  = new_cols

                y_true = None
                if roles["label"]:
                    roles["label"].seek(0)
                    ldata = np.load(roles["label"], allow_pickle=True)
                    # Handle 2D label arrays (e.g. shape (N, 25)) —
                    # any channel flagged = anomaly, collapse to 1D
                    if ldata.ndim == 2:
                        ldata = ldata.max(axis=1)
                    elif ldata.ndim > 2:
                        ldata = ldata.reshape(-1)
                    y_true = pd.Series(ldata.astype(int))

                X_train = preprocess(train_raw, is_timeseries=True)
                X_test  = preprocess(test_raw,  is_timeseries=True)

                # 🔥 KEEP COLUMN CONSISTENCY — guard against preprocess changing col count
                if len(X_train.columns) == len(new_cols):
                    X_train.columns = new_cols
                if len(X_test.columns) == len(new_cols):
                    X_test.columns  = new_cols

                # 🔥 FIX: y_train must match X_train row count (train set, NOT label file)
                # Labels come with the test set for MSL/SMAP — supervised models need
                # a y_train aligned to X_train, so we use the test labels truncated/tiled
                # to match training size, or set None to skip supervised models gracefully.
                if y_true is not None:
                    n_train = len(X_train)
                    n_labels = len(y_true)
                    if n_labels >= n_train:
                        # Labels longer than train — take first n_train rows
                        y_train = y_true.values[:n_train]
                    else:
                        # Labels shorter — pad by repeating (rare edge case)
                        import math
                        repeats = math.ceil(n_train / n_labels)
                        y_train = np.tile(y_true.values, repeats)[:n_train]
                else:
                    y_train = None

                base_df = test_raw.copy().reset_index(drop=True)

            # ===============================
            # CASE 2: Single file
            # ===============================
            else:
                single = roles["single"] or uploaded_files[0]
                raw_df = load_file(single, single.name)

                # 🔥 AUTO SENSOR NAMING — pass filename for dataset-specific detection
                new_cols = auto_detect_sensor_names(raw_df, single.name)
                raw_df.columns = new_cols

                hint = label_col_hint.strip() or None
                label_name, y_true = detect_label_column(raw_df, hint=hint)

                feat_df = raw_df.copy()

                if label_name:
                    feat_df.drop(columns=[label_name], inplace=True)

                X_train = preprocess(feat_df, is_timeseries=is_timeseries)
                X_test  = X_train.copy()

                # 🔥 KEEP COLUMN CONSISTENT
                X_train.columns = feat_df.columns
                X_test.columns  = feat_df.columns

                y_train = y_true.values if y_true is not None else None
                base_df = raw_df.copy().reset_index(drop=True)

            # ===============================
            # CONTAMINATION
            # ===============================
            contamination = (
                auto_contamination(y_train, contamination_input)
                if auto_cont else contamination_input
            )

            st.session_state["contamination_used"] = contamination

        except Exception as e:
            st.error(f"❌ File loading error: {e}")
            st.stop()

    all_results = {}
    progress = st.progress(0, text="Initializing…")

    for i, name in enumerate(selected_models):
        is_sup = name in SUPERVISED_MODELS
        if is_sup and y_train is None:
            st.warning(f"⚠️ {name} skipped — no labels.")
            continue
        with st.spinner(f"{'Auto-tuning' if (name=='Isolation Forest' and auto_tune_if) else 'Training'} **{name}**…"):
            try:
                preds, scores, t, extra = run_model(
                    name, X_train, X_test,
                    y_train           = y_train if is_sup else None,
                    contamination     = contamination,
                    auto_tune         = (name == "Isolation Forest" and auto_tune_if),
                    y_true_for_tuning = y_true.values[:len(X_test)] if y_true is not None else None
                )
                result = {"predictions":preds, "scores":scores, "train_time":t, "extra":extra}
                if y_true is not None:
                    y_eval = y_true.values[:len(preds)]
                    best_t, best_f1, all_f1s, all_precs, all_recs = find_best_threshold(scores, y_eval)
                    result.update({
                        "best_threshold"  : best_t,
                        "best_f1"         : best_f1,
                        "threshold_f1s"   : all_f1s,
                        "threshold_precs" : all_precs,
                        "threshold_recs"  : all_recs,
                        "metrics"         : evaluate(y_eval, preds, name, t),
                        "best_metrics"    : evaluate(y_eval, predict_with_threshold(scores, best_t), name, t),
                    })
            except Exception as e:
                st.error(f"❌ {name} failed: {e}")
                progress.progress((i+1)/len(selected_models))
                continue
        all_results[name] = result
        progress.progress((i+1)/len(selected_models), text=f"✓ {name}")

    progress.empty()
    if not all_results:
        st.error("No models ran successfully.")
        st.stop()

    st.session_state.update({
        "all_results": all_results,
        "base_df"    : base_df,
        "y_true"     : y_true,
        "ready"      : True
    })

# ══════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════
if not st.session_state.get("ready"):
    st.stop()

all_results = st.session_state["all_results"]
base_df     = st.session_state["base_df"]
y_true      = st.session_state["y_true"]
model_names = list(all_results.keys())
cont_used   = st.session_state.get("contamination_used", 0.1)

st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Detection Results</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="cont-banner">
  ⚙️ &nbsp;Contamination: <span class="cv">{cont_used}</span>
  &nbsp;·&nbsp; {round(cont_used*100,1)}% anomaly rate assumed
</div>
""", unsafe_allow_html=True)

sc1, sc2 = st.columns([3, 1])
with sc1:
    active_model = st.selectbox("Model", model_names, label_visibility="collapsed")
with sc2:
    btype = "Supervised" if active_model in SUPERVISED_MODELS else "Unsupervised"
    bcls  = "b-green" if btype == "Supervised" else "b-grad"
    st.markdown(f'<div style="padding-top:10px"><span class="badge {bcls}">{btype}</span></div>', unsafe_allow_html=True)

res    = all_results[active_model]
scores = res["scores"]

if use_manual_threshold:
    preds       = predict_with_threshold(scores, manual_threshold)
    thresh_used = manual_threshold
    thresh_mode = f"Manual — {manual_threshold}%"
elif use_auto_threshold and "best_threshold" in res:
    preds       = predict_with_threshold(scores, res["best_threshold"])
    thresh_used = res["best_threshold"]
    thresh_mode = f"Auto-optimized — {res['best_threshold']}%"
else:
    preds       = res["predictions"]
    thresh_used = None
    thresh_mode = "Model default"

total = len(preds)
n_an  = int(preds.sum())
n_no  = total - n_an
rate  = round(n_an / total * 100, 2)
extra = res.get("extra", {})

if extra.get("tuned"):
    cfg = extra["best_config"]
    st.markdown(f"""
    <div class="success-banner">✅ &nbsp;<strong>Isolation Forest Auto-Tuned</strong> —
    n_estimators=<strong>{cfg['n_estimators']}</strong> ·
    max_samples=<strong>{cfg['max_samples']}</strong> ·
    max_features=<strong>{cfg['max_features']}</strong>
    → Best F1: <strong>{extra['best_f1']}</strong></div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="tune-banner">
  🎯 &nbsp;<strong>Threshold:</strong>&nbsp;{thresh_mode}
  &nbsp;&nbsp;<span class="badge b-red">{n_an:,} anomalies</span>
  <span class="badge b-amber">{rate}%</span>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
for col, (val, lbl, color, ico) in zip([m1,m2,m3,m4], [
    (f"{total:,}", "Total Records", G2,    "🗃️"),
    (f"{n_no:,}",  "Normal Points", GREEN, "✅"),
    (f"{n_an:,}",  "Anomalies",     RED,   "⚠️"),
    (f"{rate}%",   "Anomaly Rate",  AMBER, "📊"),
]):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-glow" style="background:{color}"></div>
          <div class="metric-ico">{ico}</div>
          <div class="metric-val" style="color:{color}">{val}</div>
          <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Evaluation ───────────────────────────────────────
if y_true is not None:
    y_eval  = y_true.values[:len(preds)]
    metrics = evaluate(y_eval, preds, active_model, res["train_time"])
    st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Model Evaluation</div>', unsafe_allow_html=True)
    e1,e2,e3,e4,e5 = st.columns(5)
    for col, (key, lbl) in zip([e1,e2,e3,e4,e5], [
        ("accuracy","Accuracy"),("precision","Precision"),
        ("recall","Recall"),("f1_score","F1 Score"),("train_time","Train Time")
    ]):
        with col:
            v = metrics[key]
            if key == "train_time": color, disp = TEXT_SEC, f"{v}s"
            else: color = (GREEN if v >= 0.7 else AMBER if v >= 0.5 else RED); disp = str(v)
            st.markdown(f'<div class="eval-card"><div class="eval-val" style="color:{color}">{disp}</div><div class="eval-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ── Threshold chart ───────────────────────────────────
if "threshold_f1s" in res and y_true is not None:
    st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Threshold Optimization</div>', unsafe_allow_html=True)
    thresholds = list(range(1, 51))
    fig, ax    = plt.subplots(figsize=(10, 3.8), facecolor=PLOT_BG)
    sfig(fig, [ax])
    ax.fill_between(thresholds, res["threshold_f1s"], alpha=0.18, color=G2)
    ax.plot(thresholds, res["threshold_f1s"],   color=G2,    lw=2.5, label="F1 Score")
    ax.plot(thresholds, res["threshold_precs"], color=GREEN, lw=2,   label="Precision", ls="--")
    ax.plot(thresholds, res["threshold_recs"],  color=RED,   lw=2,   label="Recall",    ls=":")
    best_t = res.get("best_threshold")
    if thresh_used:
        ax.axvline(thresh_used, color=AMBER, lw=1.8, ls="--", label=f"Current ({thresh_used}%)", alpha=.9)
    if best_t:
        ax.axvline(best_t, color=G3, lw=1.8, ls=":", label=f"Optimal ({best_t}%)", alpha=.9)
        opt_f1 = res["threshold_f1s"][best_t-1] if best_t <= len(res["threshold_f1s"]) else None
        if opt_f1:
            ax.scatter([best_t],[opt_f1], color=G3, s=60, zorder=5)
            ax.annotate(f"  F1={opt_f1:.2f}", xy=(best_t,opt_f1), fontsize=7.5, color=G3, va='bottom')
    ax.set_xlabel("Threshold (%)", fontsize=9)
    ax.set_ylabel("Score",         fontsize=9)
    ax.set_title(f"{active_model} — Threshold vs Metrics", fontsize=10, pad=10)
    ax.legend(facecolor=CARD, labelcolor=TEXT_PRI, fontsize=8, framealpha=.9, edgecolor=BORDER)
    ax.grid(axis="y", color=GRID_C, lw=.5, alpha=.65)
    ax.set_xlim(1, 50); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    glass_chart(fig)
    if best_t:
        bm = res.get("best_metrics", {})
        st.markdown(f'<div class="success-banner">🏆 &nbsp;<strong>Optimal threshold: {best_t}%</strong> — F1=<strong>{bm.get("f1_score","?")}</strong> | Precision=<strong>{bm.get("precision","?")}</strong> | Recall=<strong>{bm.get("recall","?")}</strong></div>', unsafe_allow_html=True)

# ── Confusion matrix + distribution ──────────────────
# FIX 5: Only create 2 columns when y_true exists, otherwise use full width
if y_true is not None:
    v1, v2 = st.columns(2)
    cm_col, dist_col = v1, v2
else:
    cm_col  = None
    dist_col = st

if y_true is not None and cm_col is not None:
    with cm_col:
        st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Confusion Matrix</div>', unsafe_allow_html=True)
        y_eval = y_true.values[:len(preds)]

        # FIX 2: labels=[0,1] ensures 2x2 matrix even if preds is all one class
        cm = confusion_matrix(y_eval, preds, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5, 4), facecolor=PLOT_BG)
        sfig(fig, [ax])
        im = ax.imshow(cm, cmap=GRAD_CMAP, vmin=0, alpha=0.9)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Normal","Anomaly"], color=TEXT_SEC, fontsize=9)
        ax.set_yticklabels(["Normal","Anomaly"], color=TEXT_SEC, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual",    fontsize=9)
        ax.set_title("Confusion Matrix", fontsize=10, pad=10)
        cm_labels = [["TN","FP"],["FN","TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i][j]:,}\n{cm_labels[i][j]}",
                        ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold", linespacing=1.6)
        cbar = plt.colorbar(im, ax=ax, fraction=.04, pad=.04)
        cbar.ax.tick_params(colors=TEXT_SEC, labelsize=7)
        plt.tight_layout()
        glass_chart(fig)

        # FIX 2 (cont): safe unpack now that matrix is guaranteed 2x2
        tn, fp, fn, tp = cm.ravel()
        precision_cm   = tp / (tp + fp + 1e-9)
        recall_cm      = tp / (tp + fn + 1e-9)
        specificity_cm = tn / (tn + fp + 1e-9)
        fpr_cm         = fp / (fp + tn + 1e-9)

        if precision_cm > 0.8 and recall_cm > 0.8:
            interp = "🔥 Excellent — strong detection with low errors."
        elif precision_cm > 0.6:
            interp = "⚖️ Balanced — some threshold tuning recommended."
        else:
            interp = "⚠️ Needs improvement — high misclassification detected."

        st.markdown(f"""
        <div class="info-banner">
        <b>📊 Confusion Matrix Insight</b><br><br>
        • <b>TN:</b> {tn:,} → Normal correctly identified<br>
        • <b>FP:</b> {fp:,} → Normal wrongly flagged ⚠️<br>
        • <b>FN:</b> {fn:,} → Missed anomalies ❌<br>
        • <b>TP:</b> {tp:,} → Anomalies correctly caught 🎯<br><br>
        <b>🔍 Derived Metrics:</b><br>
        Precision: {precision_cm:.3f} · Recall: {recall_cm:.3f} · Specificity: {specificity_cm:.3f} · FPR: {fpr_cm:.3f}<br><br>
        <b>💡 Interpretation:</b> {interp}
        </div>
        """, unsafe_allow_html=True)

with dist_col:
    st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Anomaly Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=PLOT_BG)
    sfig(fig, [ax])
    wedges, _, autos = ax.pie(
        [n_no, n_an], labels=["Normal","Anomaly"],
        colors=[GREEN, G3], autopct="%1.1f%%", startangle=90, pctdistance=0.72,
        wedgeprops=dict(linewidth=2.5, edgecolor=PLOT_BG, width=0.55),
        textprops={"color":TEXT_PRI,"fontsize":9}
    )
    for a in autos:
        a.set_fontsize(9); a.set_color(TEXT_PRI)
    ax.text(0, 0, f"{rate}%\nanom.", ha="center", va="center",
            color=G3, fontsize=10, fontweight="bold")
    ax.set_title("Normal vs Anomaly Split", fontsize=10, pad=10)
    plt.tight_layout()
    glass_chart(fig)

# ── Score charts ─────────────────────────────────────
st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Anomaly Score Distribution</div>', unsafe_allow_html=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5), facecolor=PLOT_BG)
sfig(fig, [ax1, ax2])
ax1.hist(scores[preds==0], bins=60, alpha=0.75, color=GREEN, label="Normal",  edgecolor=PLOT_BG, lw=.3)
ax1.hist(scores[preds==1], bins=60, alpha=0.75, color=G3,    label="Anomaly", edgecolor=PLOT_BG, lw=.3)
if thresh_used:
    cutoff = np.percentile(scores, 100-thresh_used)
    ax1.axvline(cutoff, color=AMBER, lw=2, ls="--", label=f"Threshold ({thresh_used}%)")
ax1.set_xlabel("Anomaly Score", fontsize=9); ax1.set_ylabel("Count", fontsize=9)
ax1.set_title("Score Histogram", fontsize=10, pad=8)
ax1.grid(axis="y", color=GRID_C, lw=.5, alpha=.6)
ax1.legend(facecolor=CARD, labelcolor=TEXT_PRI, fontsize=8, framealpha=.9, edgecolor=BORDER)
sorted_scores = np.sort(scores)
cumulative    = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
ax2.plot(sorted_scores, cumulative, color=G2, lw=2.5, label="CDF")
if thresh_used:
    cutoff2 = np.percentile(scores, 100-thresh_used)
    ax2.axvline(cutoff2, color=AMBER, lw=2, ls="--", label=f"Threshold ({thresh_used}%)")
    ax2.fill_betweenx([0,1], cutoff2, sorted_scores.max(), alpha=.1, color=RED)
ax2.set_xlabel("Anomaly Score", fontsize=9); ax2.set_ylabel("Cumulative Fraction", fontsize=9)
ax2.set_title("Cumulative Distribution (CDF)", fontsize=10, pad=8)
ax2.set_ylim(0, 1); ax2.grid(color=GRID_C, lw=.5, alpha=.6)
ax2.legend(facecolor=CARD, labelcolor=TEXT_PRI, fontsize=8, framealpha=.9, edgecolor=BORDER)
plt.tight_layout()
glass_chart(fig)

# ── Score timeline ────────────────────────────────────
st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Score Timeline</div>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 3), facecolor=PLOT_BG)
sfig(fig, [ax])
idx = np.arange(len(scores))
ax.plot(idx, scores, color=G2, lw=1.2, alpha=.8, label="Anomaly Score")
if n_an > 0:
    ax.scatter(idx[preds==1], scores[preds==1], color=RED, s=12, zorder=4, alpha=.85, label=f"Anomaly ({n_an:,})")
if thresh_used:
    cutoff = np.percentile(scores, 100-thresh_used)
    ax.axhline(cutoff, color=AMBER, lw=1.5, ls="--", label=f"Threshold ({thresh_used}%)", alpha=.9)
    ax.fill_between(idx, cutoff, scores.max()*1.05, where=(scores>=cutoff), alpha=.08, color=RED)
ax.set_xlabel("Record Index", fontsize=9); ax.set_ylabel("Anomaly Score", fontsize=9)
ax.set_title(f"{active_model} — Score Timeline", fontsize=10, pad=8)
ax.grid(color=GRID_C, lw=.5, alpha=.5)
ax.legend(facecolor=CARD, labelcolor=TEXT_PRI, fontsize=8, framealpha=.9, edgecolor=BORDER)
plt.tight_layout()
glass_chart(fig)

# ── Config comparison ─────────────────────────────────
if extra.get("tuned") and extra.get("all_configs"):
    st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Isolation Forest Config Comparison</div>', unsafe_allow_html=True)
    cfg_df = pd.DataFrame(extra["all_configs"])
    cfg_df.columns = ["n_estimators","max_samples","max_features","F1 Score"]
    cfg_sorted = cfg_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

    def simple_html_table(df):
        th = f"padding:8px 14px;text-align:left;font-size:.72rem;font-weight:700;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:.8px;border-bottom:2px solid {BORDER2};background:{CARD2};white-space:nowrap"
        td = f"padding:7px 14px;font-size:.8rem;color:{TEXT_PRI};border-bottom:1px solid {BORDER};font-family:'Fira Code',monospace"
        td_best = f"padding:7px 14px;font-size:.8rem;color:{GREEN};border-bottom:1px solid {BORDER};font-family:'Fira Code',monospace;font-weight:700"
        wrap = f"overflow-x:auto;border-radius:12px;border:1px solid {BORDER};box-shadow:{SHADOW}"
        html = f'<div style="{wrap}"><table style="width:100%;border-collapse:collapse;background:{CARD}">'
        html += "<thead><tr>" + "".join(f'<th style="{th}">{c}</th>' for c in df.columns) + "</tr></thead><tbody>"
        for i, (_, row) in enumerate(df.iterrows()):
            bg = f"background:{GREEN}10" if i == 0 else f"background:{CARD}"
            html += f'<tr style="{bg}">'
            for j, val in enumerate(row):
                style = td_best if (i == 0 and j == len(row)-1) else td
                html += f'<td style="{style}">{val}</td>'
            html += "</tr>"
        html += "</tbody></table></div>"
        return html

    st.markdown(simple_html_table(cfg_sorted), unsafe_allow_html=True)

# ── Model comparison ──────────────────────────────────
if len(all_results) > 1 and any("metrics" in v for v in all_results.values()):
    st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Model Comparison</div>', unsafe_allow_html=True)
    rows = []
    for name, r in all_results.items():
        if "metrics" in r:
            m  = r["metrics"]
            bm = r.get("best_metrics", m)
            rows.append({
                "Model"          : name,
                "Default F1"     : m["f1_score"],
                "Best F1"        : bm["f1_score"],
                "Best Thresh (%)" : r.get("best_threshold", "-"),
                "Precision"      : bm["precision"],
                "Recall"         : bm["recall"],
                "Train Time"     : f"{m['train_time']}s",
                "Type"           : "Supervised" if name in SUPERVISED_MODELS else "Unsupervised"
            })
    if rows:
        comp_df   = pd.DataFrame(rows).sort_values("Best F1", ascending=False).reset_index(drop=True)
        best_name = comp_df.iloc[0]["Model"]
        best_val  = comp_df.iloc[0]["Best F1"]
        st.markdown(f'<div class="success-banner">🏆 &nbsp;<strong>Best model: {best_name}</strong> — F1 = <strong>{best_val}</strong></div>', unsafe_allow_html=True)
        def model_html_table(df):
            th = f"padding:8px 14px;text-align:left;font-size:.72rem;font-weight:700;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:.8px;border-bottom:2px solid {BORDER2};background:{CARD2};white-space:nowrap"
            td = f"padding:7px 14px;font-size:.8rem;color:{TEXT_PRI};border-bottom:1px solid {BORDER};font-family:'Fira Code',monospace"
            wrap = f"overflow-x:auto;border-radius:12px;border:1px solid {BORDER};box-shadow:{SHADOW}"
            html = f'<div style="{wrap}"><table style="width:100%;border-collapse:collapse;background:{CARD}">'
            html += "<thead><tr>" + "".join(f'<th style="{th}">{c}</th>' for c in df.columns) + "</tr></thead><tbody>"
            for i, (_, row) in enumerate(df.iterrows()):
                bg = f"background:{G1}12" if i == 0 else f"background:{CARD if i%2==0 else CARD2}"
                html += f'<tr style="{bg}">'
                for col, val in zip(df.columns, row):
                    color = TEXT_PRI
                    if col in ("Best F1","Default F1","Precision","Recall") and isinstance(val, float):
                        color = GREEN if val >= 0.7 else AMBER if val >= 0.5 else RED
                        val = f"{val:.4f}"
                    html += f'<td style="{td.replace(TEXT_PRI, color)}">{val}</td>'
                html += "</tr>"
            html += "</tbody></table></div>"
            return html
        st.markdown(model_html_table(comp_df), unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Model F1 Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(max(6, len(rows)*1.6), 3.5), facecolor=PLOT_BG)
        sfig(fig, [ax])
        names_  = comp_df["Model"].tolist()
        f1_vals = comp_df["Best F1"].tolist()
        colors_ = [G2 if i==0 else BORDER2 for i in range(len(names_))]
        bars = ax.bar(names_, f1_vals, color=colors_, width=.55, edgecolor=PLOT_BG, lw=1.5)
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    color=TEXT_PRI, fontweight="600")
        ax.set_ylabel("Best F1 Score", fontsize=9); ax.set_ylim(0, 1.12)
        ax.set_title("Model Comparison — Best Threshold F1", fontsize=10, pad=10)
        ax.grid(axis="y", color=GRID_C, lw=.5, alpha=.6)
        plt.xticks(rotation=20, ha='right', fontsize=8)
        plt.tight_layout()
        glass_chart(fig)

# ── Prediction table ──────────────────────────────────
st.markdown('<div class="sec-hdr"><div class="sec-bar"></div>Prediction Table</div>', unsafe_allow_html=True)
result_df = base_df.copy()
result_df["predicted_anomaly"] = preds
result_df["anomaly_score"]     = np.round(scores, 6)

tf1, tf2, tf3 = st.columns([2, 1, 1])
with tf1:
    filter_opt = st.selectbox("Filter", ["All records","Anomalies only","Normal only"], label_visibility="collapsed")
with tf2:
    sort_by = st.selectbox("Sort by", ["Default","Score ↓","Score ↑"], label_visibility="collapsed")
with tf3:
    st.markdown(f'<div style="padding-top:8px;font-size:.75rem;color:{TEXT_MUT}">Total: {len(result_df):,} rows</div>', unsafe_allow_html=True)

display_df = result_df.copy()
if filter_opt == "Anomalies only": display_df = display_df[display_df["predicted_anomaly"]==1]
elif filter_opt == "Normal only":  display_df = display_df[display_df["predicted_anomaly"]==0]
if sort_by == "Score ↓":  display_df = display_df.sort_values("anomaly_score", ascending=False)
elif sort_by == "Score ↑": display_df = display_df.sort_values("anomaly_score", ascending=True)

def render_html_table(df, max_rows=500):
    """Render a fully themed HTML table — works in both light and dark mode."""
    rows_to_show = df.head(max_rows)
    cols = list(rows_to_show.columns)

    th_style  = f"padding:9px 12px;text-align:left;font-size:.72rem;font-weight:700;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:.8px;border-bottom:2px solid {BORDER2};background:{CARD2};white-space:nowrap"
    td_style  = f"padding:8px 12px;font-size:.78rem;color:{TEXT_PRI};border-bottom:1px solid {BORDER};white-space:nowrap;font-family:'Fira Code',monospace"
    td_an     = f"padding:8px 12px;font-size:.78rem;color:{RED};border-bottom:1px solid {BORDER};white-space:nowrap;font-family:'Fira Code',monospace"
    row_norm  = f"background:{CARD}"
    row_anom  = f"background:{RED}12"
    wrap      = f"overflow-x:auto;border-radius:14px;border:1px solid {BORDER};box-shadow:{SHADOW};max-height:400px;overflow-y:auto"

    html = f'<div style="{wrap}"><table style="width:100%;border-collapse:collapse;background:{CARD}">'
    html += "<thead><tr>" + "".join(f'<th style="{th_style}">{c}</th>' for c in cols) + "</tr></thead>"
    html += "<tbody>"

    for _, row in rows_to_show.iterrows():
        is_anom  = int(row.get("predicted_anomaly", 0)) == 1
        row_bg   = row_anom if is_anom else row_norm
        html    += f'<tr style="{row_bg}">'
        for c in cols:
            val = row[c]
            # Format floats nicely
            if isinstance(val, float):
                val = f"{val:.6f}"
            elif isinstance(val, (int, np.integer)):
                val = str(int(val))
            else:
                val = str(val)
            # Anomaly column — show badge
            if c == "predicted_anomaly":
                if is_anom:
                    cell = f'<span style="background:{RED}22;color:{RED};padding:2px 8px;border-radius:20px;font-size:.68rem;font-weight:700">ANOMALY</span>'
                else:
                    cell = f'<span style="background:{GREEN}22;color:{GREEN};padding:2px 8px;border-radius:20px;font-size:.68rem;font-weight:700">NORMAL</span>'
                html += f'<td style="{td_style}">{cell}</td>'
            else:
                td = td_an if is_anom else td_style
                html += f'<td style="{td}">{val}</td>'
        html += "</tr>"

    html += f'</tbody></table></div>'
    return html

st.markdown(render_html_table(display_df), unsafe_allow_html=True)
if len(display_df) > 500:
    st.caption(f"Showing first 500 of {len(display_df):,} rows.")

st.markdown("<br>", unsafe_allow_html=True)
dl1, dl2 = st.columns([1, 3])
with dl1:
    export_df = result_df.copy()
    if not export_scores:
        export_df = export_df.drop(columns=["anomaly_score"], errors="ignore")
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download Results as CSV", csv,
                       "anomaly_predictions.csv", "text/csv")
with dl2:
    st.markdown(f'<div style="padding-top:8px;font-size:.75rem;color:{TEXT_MUT}">{"Includes raw anomaly scores · " if export_scores else ""}Predictions for all {len(result_df):,} records</div>', unsafe_allow_html=True)