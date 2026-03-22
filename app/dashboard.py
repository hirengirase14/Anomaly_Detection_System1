import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.loader   import load_file, detect_label_column, preprocess
from src.pipeline import (
    run_model, evaluate, predict_with_threshold,
    find_best_threshold, auto_contamination,
    UNSUPERVISED_MODELS, SUPERVISED_MODELS
)

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Anomaly Detection System", page_icon="🔬", layout="wide")

# ══════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background:#0d1117; color:#e6edf3; }
.main-title { font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:600; color:#58a6ff; letter-spacing:-1px; }
.sub-title  { font-size:0.9rem; color:#8b949e; margin-top:4px; }
.sec-header { font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#8b949e; text-transform:uppercase;
              letter-spacing:2px; border-bottom:1px solid #21262d; padding-bottom:8px; margin:28px 0 16px 0; }
.card       { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px 22px; text-align:center; }
.val        { font-family:'IBM Plex Mono',monospace; font-size:1.8rem; font-weight:600; color:#58a6ff; }
.lbl        { font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
.info-box   { background:#161b22; border-left:3px solid #58a6ff; border-radius:0 8px 8px 0;
              padding:10px 14px; font-size:0.85rem; color:#8b949e; margin:10px 0; }
.success-box { background:#0d2818; border:1px solid #3fb950; border-radius:8px;
               padding:12px 16px; font-size:0.88rem; color:#3fb950; margin:10px 0; }
.tune-box   { background:#161b22; border:1px solid #d29922; border-radius:10px; padding:16px 20px; margin:12px 0; }
div[data-testid="stFileUploader"] { border:2px dashed #30363d; border-radius:12px; padding:6px; background:#161b22; }
div[data-testid="stFileUploader"]:hover { border-color:#58a6ff; }
.stButton>button { background:#238636; color:white; border:none; border-radius:6px;
                   font-family:'IBM Plex Mono',monospace; font-size:0.85rem; padding:10px 20px; width:100%; }
.stButton>button:hover { background:#2ea043; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════
st.markdown('<div class="main-title">🔬 Machine Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload any dataset — auto-tune models — get the most accurate results</div>', unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Upload Dataset")
    st.markdown('<div class="info-box">Supports: CSV, Excel, JSON, NPY, NPZ, Parquet, TSV, TXT, HDF5<br><br>For multi-file datasets (like MSL), upload all files together.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & Drop files here",
        type=["csv","xlsx","xls","json","npy","npz","parquet","tsv","txt","h5","hdf5"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    contamination_input = st.slider(
        "Anomaly Rate (manual)",
        0.01, 0.5, 0.1, 0.01,
        help="Used only if auto-contamination is disabled."
    )

    auto_cont = st.checkbox(
        "Auto-detect contamination from labels",
        value=True,
        help="If labels are available, automatically calculate the exact anomaly ratio."
    )

    is_timeseries = st.checkbox(
        "Time Series Data", value=False,
        help="Enable for sensor/telemetry data."
    )

    label_col_hint = st.text_input(
        "Label column name (optional)",
        placeholder="e.g. faulty, label, anomaly"
    )

    st.markdown("---")
    st.markdown("### 🎯 Threshold Settings")
    st.markdown('<div class="info-box">Auto-threshold scans all values and picks the one with the highest F1 Score automatically.</div>', unsafe_allow_html=True)

    use_auto_threshold = st.checkbox(
        "Auto-find best threshold",
        value=True,
        help="Automatically finds the threshold that gives highest F1 score."
    )

    use_manual_threshold = st.checkbox(
        "Manual threshold override",
        value=False,
        help="Manually set threshold after auto-detection."
    )

    manual_threshold = st.slider(
        "Manual Threshold (%)",
        1, 50, 10, 1,
        disabled=not use_manual_threshold,
        help="Top X% of scores flagged as anomalies."
    )

    st.markdown("---")
    st.markdown("### 🤖 Select Models")
    st.markdown('<div class="info-box">Unsupervised: no labels needed<br>Supervised: requires label column</div>', unsafe_allow_html=True)

    auto_tune_if = st.checkbox(
        "🔧 Auto-tune Isolation Forest",
        value=True,
        help="Tests 7 configurations and picks the best one automatically."
    )

    st.markdown("**Unsupervised**")
    selected_unsupervised = []
    for name, desc in UNSUPERVISED_MODELS.items():
        if st.checkbox(name, value=(name == "Isolation Forest"), key=f"un_{name}", help=desc):
            selected_unsupervised.append(name)

    st.markdown("**Supervised** *(needs labels)*")
    selected_supervised = []
    for name, desc in SUPERVISED_MODELS.items():
        if st.checkbox(name, value=False, key=f"su_{name}", help=desc):
            selected_supervised.append(name)

    run_btn = st.button("▶ Run Detection")

# ══════════════════════════════════════════════════════
#  WAITING STATE
# ══════════════════════════════════════════════════════
if not uploaded_files:
    c1, c2, c3 = st.columns(3)
    for col, icon, step, desc in zip(
        [c1,c2,c3], ["📁","⚙️","🚀"],
        ["Step 1","Step 2","Step 3"],
        ["Upload dataset file(s) from the sidebar",
         "Enable auto-tune and auto-threshold",
         "Click Run Detection for best results"]
    ):
        with col:
            st.markdown(
                f'<div class="card"><div style="font-size:2rem">{icon}</div>'
                f'<div style="font-weight:600;margin-top:8px">{step}</div>'
                f'<div style="color:#8b949e;font-size:0.82rem;margin-top:4px">{desc}</div></div>',
                unsafe_allow_html=True
            )
    st.stop()

# ══════════════════════════════════════════════════════
#  UPLOADED FILES
# ══════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Uploaded Files</div>', unsafe_allow_html=True)
fc = st.columns(max(len(uploaded_files), 1))
for i, f in enumerate(uploaded_files):
    with fc[i]:
        st.markdown(
            f'<div class="card"><div style="font-size:1.4rem">📄</div>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#58a6ff;margin-top:6px">{f.name}</div>'
            f'<div style="color:#8b949e;font-size:0.75rem">{round(f.size/1024,1)} KB</div></div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════
#  FILE ROLE DETECTION
# ══════════════════════════════════════════════════════
def identify_roles(files):
    roles = {"train": None, "test": None, "label": None, "single": None}
    if len(files) == 1:
        roles["single"] = files[0]
        return roles
    for f in files:
        n = f.name.lower()
        if "train" in n:   roles["train"] = f
        elif "label" in n: roles["label"] = f
        elif "test"  in n: roles["test"]  = f
    if not roles["train"] and not roles["test"]:
        roles["single"] = files[0]
    return roles

# ══════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════
if run_btn:
    selected_models = selected_unsupervised + selected_supervised
    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    with st.spinner("Loading and processing files..."):
        try:
            roles = identify_roles(uploaded_files)

            if roles["train"] and roles["test"]:
                train_raw = load_file(roles["train"], roles["train"].name)
                test_raw  = load_file(roles["test"],  roles["test"].name)
                y_true = None
                if roles["label"]:
                    ldata  = np.load(roles["label"], allow_pickle=True)
                    y_true = pd.Series(ldata.astype(int))
                X_train = preprocess(train_raw, is_timeseries=True)
                X_test  = preprocess(test_raw,  is_timeseries=True)
                y_train = y_true.values[:len(X_train)] if y_true is not None else None
                base_df = test_raw.copy().reset_index(drop=True)
            else:
                single = roles["single"] or uploaded_files[0]
                raw_df = load_file(single, single.name)
                hint   = label_col_hint.strip() or None
                label_name, y_true = detect_label_column(raw_df, hint=hint)
                feat_df = raw_df.copy()
                if label_name:
                    feat_df.drop(columns=[label_name], inplace=True)
                X_train = preprocess(feat_df, is_timeseries=is_timeseries)
                X_test  = X_train.copy()
                y_train = y_true.values if y_true is not None else None
                base_df = raw_df.copy().reset_index(drop=True)

            # Auto contamination
            contamination = auto_contamination(y_train, contamination_input) if auto_cont else contamination_input
            st.session_state["contamination_used"] = contamination

        except Exception as e:
            st.error(f"❌ File loading error: {e}")
            st.stop()

    all_results = {}
    progress    = st.progress(0, text="Running models...")

    for i, name in enumerate(selected_models):
        is_supervised = name in SUPERVISED_MODELS
        if is_supervised and y_train is None:
            st.warning(f"⚠️ **{name}** skipped — no label column found.")
            continue

        with st.spinner(f"{'🔧 Auto-tuning' if (name == 'Isolation Forest' and auto_tune_if) else 'Training'} {name}..."):
            try:
                preds, scores, t, extra = run_model(
                    name, X_train, X_test,
                    y_train              = y_train if is_supervised else None,
                    contamination        = contamination,
                    auto_tune            = (name == "Isolation Forest" and auto_tune_if),
                    y_true_for_tuning    = y_true.values[:len(X_test)] if y_true is not None else None
                )

                result = {
                    "predictions" : preds,
                    "scores"      : scores,
                    "train_time"  : t,
                    "extra"       : extra
                }

                # Auto-find best threshold
                if y_true is not None:
                    y_eval = y_true.values[:len(preds)]
                    best_t, best_f1, all_f1s, all_precs, all_recs = find_best_threshold(scores, y_eval)
                    result["best_threshold"]  = best_t
                    result["best_f1"]         = best_f1
                    result["threshold_f1s"]   = all_f1s
                    result["threshold_precs"] = all_precs
                    result["threshold_recs"]  = all_recs
                    result["metrics"]         = evaluate(y_eval, preds, name, t)
                    # Also compute metrics at best threshold
                    best_preds = predict_with_threshold(scores, best_t)
                    result["best_metrics"] = evaluate(y_eval, best_preds, name, t)

            except Exception as e:
                st.error(f"❌ {name} failed: {e}")
                progress.progress((i+1)/len(selected_models))
                continue

        all_results[name] = result
        progress.progress((i+1)/len(selected_models), text=f"Completed: {name}")

    progress.empty()
    if not all_results:
        st.error("No models ran successfully.")
        st.stop()

    st.session_state["all_results"] = all_results
    st.session_state["base_df"]     = base_df
    st.session_state["y_true"]      = y_true
    st.session_state["ready"]       = True

# ══════════════════════════════════════════════════════
#  DISPLAY RESULTS
# ══════════════════════════════════════════════════════
if not st.session_state.get("ready"):
    st.stop()

all_results = st.session_state["all_results"]
base_df     = st.session_state["base_df"]
y_true      = st.session_state["y_true"]
model_names = list(all_results.keys())
cont_used   = st.session_state.get("contamination_used", 0.1)

st.markdown('<div class="sec-header">Results</div>', unsafe_allow_html=True)

# Show contamination used
st.markdown(
    f'<div class="info-box">⚙️ Contamination used: <b style="color:#58a6ff">{cont_used}</b> '
    f'({round(cont_used*100,1)}% anomaly rate assumed)</div>',
    unsafe_allow_html=True
)

active_model = st.selectbox("View results for model:", model_names)
res    = all_results[active_model]
scores = res["scores"]

# ── THRESHOLD DECISION ────────────────────────────────
if use_manual_threshold:
    preds      = predict_with_threshold(scores, manual_threshold)
    thresh_used = manual_threshold
    thresh_mode = f"Manual ({manual_threshold}%)"
elif use_auto_threshold and "best_threshold" in res:
    preds      = predict_with_threshold(scores, res["best_threshold"])
    thresh_used = res["best_threshold"]
    thresh_mode = f"Auto-optimized ({res['best_threshold']}%)"
else:
    preds      = res["predictions"]
    thresh_used = None
    thresh_mode = "Model default"

total = len(preds)
n_an  = int(preds.sum())
n_no  = total - n_an
rate  = round(n_an / total * 100, 2)

# ── AUTO-TUNE INFO ────────────────────────────────────
extra = res.get("extra", {})
if extra.get("tuned"):
    cfg = extra["best_config"]
    st.markdown(
        f'<div class="success-box">✅ <b>Isolation Forest Auto-Tuned</b> — '
        f'Best config: n_estimators=<b>{cfg["n_estimators"]}</b>, '
        f'max_samples=<b>{cfg["max_samples"]}</b>, '
        f'max_features=<b>{cfg["max_features"]}</b> '
        f'→ Best F1 (at optimal threshold): <b>{extra["best_f1"]}</b></div>',
        unsafe_allow_html=True
    )

# ── THRESHOLD INFO ────────────────────────────────────
st.markdown(
    f'<div class="tune-box">🎯 <b>Threshold mode:</b> {thresh_mode} — '
    f'<b style="color:#f85149">{n_an:,}</b> anomalies detected ({rate}%)</div>',
    unsafe_allow_html=True
)

# ── METRIC CARDS ─────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl, color in zip(
    [m1,m2,m3,m4],
    [f"{total:,}", f"{n_no:,}", f"{n_an:,}", f"{rate}%"],
    ["Total Records","Normal","Anomalies","Anomaly Rate"],
    ["#58a6ff","#3fb950","#f85149","#d29922"]
):
    with col:
        st.markdown(
            f'<div class="card"><div class="val" style="color:{color}">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

# ── EVALUATION ────────────────────────────────────────
if y_true is not None:
    y_eval   = y_true.values[:len(preds)]
    metrics  = evaluate(y_eval, preds, active_model, res["train_time"])

    st.markdown('<div class="sec-header">Model Evaluation</div>', unsafe_allow_html=True)

    # Show both default and best threshold metrics side by side
    if "best_metrics" in res and thresh_used:
        st.markdown(
            f'<div class="info-box">📊 Showing metrics at <b>threshold={thresh_used}%</b>. '
            f'Default model metrics shown in comparison table below.</div>',
            unsafe_allow_html=True
        )

    e1,e2,e3,e4,e5 = st.columns(5)
    for col, key, lbl in zip(
        [e1,e2,e3,e4,e5],
        ["accuracy","precision","recall","f1_score","train_time"],
        ["Accuracy","Precision","Recall","F1 Score","Train Time (s)"]
    ):
        with col:
            val = metrics[key]
            if key == "train_time":
                color, disp = "#8b949e", f"{val}s"
            else:
                color = "#3fb950" if val >= 0.7 else "#d29922" if val >= 0.5 else "#f85149"
                disp  = str(val)
            st.markdown(
                f'<div class="card"><div class="val" style="color:{color}">{disp}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )

# ── THRESHOLD IMPACT CHART ────────────────────────────
if "threshold_f1s" in res and y_true is not None:
    st.markdown('<div class="sec-header">Threshold Optimization Chart</div>', unsafe_allow_html=True)

    thresholds = list(range(1, 51))
    fig, ax    = plt.subplots(figsize=(10, 4), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    ax.plot(thresholds, res["threshold_f1s"],   color="#58a6ff", linewidth=2.5, label="F1 Score")
    ax.plot(thresholds, res["threshold_precs"],  color="#3fb950", linewidth=2,   label="Precision")
    ax.plot(thresholds, res["threshold_recs"],   color="#f85149", linewidth=2,   label="Recall")

    if thresh_used:
        ax.axvline(thresh_used, color="#d29922", linestyle="--",
                   linewidth=2, label=f"Current ({thresh_used}%)")

    best_t = res.get("best_threshold")
    if best_t:
        ax.axvline(best_t, color="#8957e5", linestyle=":",
                   linewidth=2, label=f"Optimal ({best_t}%)")

    ax.set_xlabel("Threshold (%)", color="#8b949e")
    ax.set_ylabel("Score",         color="#8b949e")
    ax.set_title(f"{active_model} — Threshold vs Metrics", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if best_t:
        bm = res.get("best_metrics", {})
        st.markdown(
            f'<div class="success-box">🏆 <b>Optimal threshold: {best_t}%</b> — '
            f'F1={bm.get("f1_score","?")} | '
            f'Precision={bm.get("precision","?")} | '
            f'Recall={bm.get("recall","?")}<br>'
            f'Set the threshold slider to <b>{best_t}</b> for best results.</div>',
            unsafe_allow_html=True
        )

# ── ISOLATION FOREST CONFIG COMPARISON ───────────────
if extra.get("tuned") and extra.get("all_configs"):
    st.markdown('<div class="sec-header">Isolation Forest Config Comparison</div>', unsafe_allow_html=True)
    cfg_df = pd.DataFrame(extra["all_configs"])
    cfg_df.columns = ["n_estimators", "max_samples", "max_features", "F1 Score"]
    cfg_df = cfg_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)

# ── MODEL COMPARISON TABLE ───────────────────────────
if len(all_results) > 1 and any("metrics" in v for v in all_results.values()):
    st.markdown('<div class="sec-header">Model Comparison</div>', unsafe_allow_html=True)
    rows = []
    for name, r in all_results.items():
        if "metrics" in r:
            m  = r["metrics"]
            bm = r.get("best_metrics", m)
            rows.append({
                "Model"              : name,
                "Default F1"         : m["f1_score"],
                "Best Threshold F1"  : bm["f1_score"],
                "Best Threshold (%)" : r.get("best_threshold", "-"),
                "Precision"          : bm["precision"],
                "Recall"             : bm["recall"],
                "Train Time"         : f"{m['train_time']}s",
                "Type"               : "Supervised" if name in SUPERVISED_MODELS else "Unsupervised"
            })

    if rows:
        comp_df   = pd.DataFrame(rows).sort_values("Best Threshold F1", ascending=False).reset_index(drop=True)
        best_name = comp_df.iloc[0]["Model"]
        best_val  = comp_df.iloc[0]["Best Threshold F1"]
        st.markdown(
            f'<div class="success-box">🏆 Best model: <b>{best_name}</b> — '
            f'F1 = <b>{best_val}</b> (at optimal threshold)</div>',
            unsafe_allow_html=True
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ── CHARTS ───────────────────────────────────────────
viz1, viz2 = st.columns(2)

if y_true is not None:
    with viz1:
        st.markdown('<div class="sec-header">Confusion Matrix</div>', unsafe_allow_html=True)
        y_eval = y_true.values[:len(preds)]
        cm     = confusion_matrix(y_eval, preds)
        fig, ax = plt.subplots(figsize=(4,3), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Normal","Anomaly"], color="#8b949e")
        ax.set_yticklabels(["Normal","Anomaly"], color="#8b949e")
        ax.set_xlabel("Predicted", color="#8b949e")
        ax.set_ylabel("Actual",    color="#8b949e")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                        color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with viz2:
    st.markdown('<div class="sec-header">Anomaly Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4,3), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    ax.pie([n_no, n_an], labels=["Normal","Anomaly"],
           colors=["#3fb950","#f85149"], autopct="%1.1f%%",
           startangle=90, textprops={"color":"#e6edf3"})
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Score distribution
st.markdown('<div class="sec-header">Score Distribution</div>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10,3), facecolor="#161b22")
ax.set_facecolor("#161b22")
ax.hist(scores[preds==0], bins=50, alpha=0.7, color="#3fb950", label="Normal")
ax.hist(scores[preds==1], bins=50, alpha=0.7, color="#f85149", label="Anomaly")
if thresh_used:
    cutoff = np.percentile(scores, 100 - thresh_used)
    ax.axvline(cutoff, color="#d29922", linestyle="--", linewidth=2,
               label=f"Threshold ({thresh_used}%)")
ax.set_xlabel("Anomaly Score", color="#8b949e")
ax.set_ylabel("Count",         color="#8b949e")
ax.set_title(f"{active_model} — Score Distribution", color="#e6edf3")
ax.tick_params(colors="#8b949e")
ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── PREDICTION TABLE ─────────────────────────────────
st.markdown('<div class="sec-header">Prediction Table</div>', unsafe_allow_html=True)

result_df = base_df.copy()
result_df["predicted_anomaly"] = preds
result_df["anomaly_score"]     = scores

filter_opt = st.selectbox("Filter", ["All records","Anomalies only","Normal only"],
                          label_visibility="collapsed")
display_df = result_df.copy()
if filter_opt == "Anomalies only": display_df = display_df[display_df["predicted_anomaly"]==1]
elif filter_opt == "Normal only":  display_df = display_df[display_df["predicted_anomaly"]==0]

def highlight(row):
    if row.get("predicted_anomaly", 0) == 1:
        return ["background-color:#2d0f0f;color:#f85149"] * len(row)
    return [""] * len(row)

st.dataframe(
    display_df.head(500).style.apply(highlight, axis=1),
    use_container_width=True, height=350
)
if len(display_df) > 500:
    st.caption(f"Showing first 500 of {len(display_df):,} rows.")

csv = result_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Results as CSV", csv,
                   "anomaly_predictions.csv", "text/csv")