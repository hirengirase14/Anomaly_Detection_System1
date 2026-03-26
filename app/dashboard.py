import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys, os, base64

from src.loader import auto_detect_sensor_names
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.loader   import load_file, detect_label_column, preprocess
from src.pipeline import (
    run_model, evaluate, predict_with_threshold,
    find_best_threshold, auto_contamination,
    UNSUPERVISED_MODELS, SUPERVISED_MODELS
)

# ── Icon loading (safe) ───────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
icon_path  = os.path.join(SCRIPT_DIR, "detection.png")
icon       = "🔮"
ICON_HTML  = '<span style="font-size:{FSIZE}">🔮</span>'

try:
    from PIL import Image as _PIL_Image
    _img = _PIL_Image.open(icon_path)
    with open(icon_path, "rb") as _f:
        _b64 = base64.b64encode(_f.read()).decode()
    icon = _img
    ICON_HTML = f'<img src="data:image/png;base64,{_b64}" style="width:{{SIZE}}px;height:{{SIZE}}px;border-radius:{{RADIUS}};object-fit:cover">'
except:
    pass

def icon_html(size=38, radius=11):
    if "{SIZE}" in ICON_HTML:
        return ICON_HTML.replace("{SIZE}", str(size)).replace("{RADIUS}", f"{radius}px")
    return ICON_HTML.replace("{FSIZE}", f"{size*0.6:.0f}px")

# Page config
st.set_page_config(
    page_title="Anomaly Detector",
    page_icon=icon,
    layout="wide"
)

# Theme
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

dark = st.session_state["dark_mode"]

# Colors
G1, G2, G3, G4 = "#7c3aed", "#a855f7", "#ec4899", "#f43f5e"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED   = "#f43f5e"
CYAN  = "#06b6d4"

BG = "#07050f" if dark else "#f6f4ff"
TEXT_PRI = "#ede9ff" if dark else "#17103a"
TEXT_SEC = "#8478b0" if dark else "#4e4280"
CARD = "#110e1e" if dark else "#ffffff"
BORDER = "#21193a" if dark else "#ddd6ff"

# Sidebar
with st.sidebar:
    st.markdown("### 📂 Dataset")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["csv","xlsx","json","npy","parquet","tsv"],
        accept_multiple_files=True
    )

    contamination_input = st.slider("Anomaly rate", 0.01, 0.5, 0.1)
    auto_cont = st.checkbox("Auto contamination", True)

    selected_models = []
    for name in UNSUPERVISED_MODELS:
        if st.checkbox(name, value=(name=="Isolation Forest")):
            selected_models.append(name)

    run_btn = st.button("🚀 Run Detection")

# Helper
def identify_roles(files):
    roles = {"train":None,"test":None,"label":None,"single":None}
    if len(files) == 1:
        roles["single"] = files[0]
        return roles
    for f in files:
        n = f.name.lower()
        if "train" in n: roles["train"] = f
        elif "test" in n: roles["test"] = f
        elif "label" in n: roles["label"] = f
    return roles

# Main logic
if run_btn:
    if not selected_models:
        st.warning("Select at least one model")
        st.stop()

    roles = identify_roles(uploaded_files)

    try:
        if roles["train"] and roles["test"]:
            train_raw = load_file(roles["train"], roles["train"].name)
            test_raw  = load_file(roles["test"], roles["test"].name)

            new_cols = auto_detect_sensor_names(train_raw)
            train_raw.columns = new_cols
            test_raw.columns  = new_cols

            X_train = preprocess(train_raw)
            X_test  = preprocess(test_raw)

            y_train = None

        else:
            raw_df = load_file(uploaded_files[0], uploaded_files[0].name)

            new_cols = auto_detect_sensor_names(raw_df)
            raw_df.columns = new_cols

            label_name, y_true = detect_label_column(raw_df)
            if label_name:
                raw_df.drop(columns=[label_name], inplace=True)

            X_train = preprocess(raw_df)
            X_test  = X_train.copy()
            y_train = y_true.values if y_true is not None else None

        contamination = auto_contamination(y_train, contamination_input) if auto_cont else contamination_input

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    for name in selected_models:
        preds, scores, t, _ = run_model(
            name,
            X_train,
            X_test,
            y_train=y_train,
            contamination=contamination
        )
        st.success(f"{name} completed ✅")