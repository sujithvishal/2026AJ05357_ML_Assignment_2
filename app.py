"""
Tumour Triage Bench — Streamlit dashboard for ML Assignment 2.

Scores a test CSV with five pre-trained, GridSearchCV-tuned scikit-learn pipelines
(Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, RBF SVM) on the
Breast Cancer Wisconsin (Diagnostic) data. Nothing is retrained here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

st.set_page_config(page_title="Tumour Triage Bench", page_icon="🩺", layout="wide")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"
FULL_DATA_CSV = HERE / "data" / "breast_cancer_wisconsin_full.csv"

LABEL = "target"                         # 0 = Malignant, 1 = Benign (positive class)
OUTCOMES = ["Malignant", "Benign"]
INPUT_COLUMNS = [c for c in pd.read_csv(FULL_DATA_CSV, nrows=0).columns if c != LABEL]
SCORE_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

PIPELINE_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

UCI_LINK = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
ROSE, TEAL, SLATE = "#c2255c", "#0b7285", "#495057"
MODEL_COLOURS = dict(zip(PIPELINE_FILES, ["#0b7285", "#e8590c", "#5f3dc4", "#2b8a3e", "#c2255c"]))
HEAT = LinearSegmentedColormap.from_list("rose_teal", ["#ffffff", "#99e9f2", TEAL])

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<style>
.hero {{
    background: linear-gradient(120deg, {TEAL} 0%, #1864ab 100%);
    color: #fff; padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;
}}
.hero h1 {{ color: #fff; margin: 0 0 .25rem 0; font-size: 2rem; }}
.hero p  {{ color: #e3fafc; margin: 0; font-size: 1rem; }}
.chip {{
    display: inline-block; padding: .15rem .7rem; margin: .5rem .4rem 0 0;
    border-radius: 999px; background: rgba(255,255,255,.18); font-size: .8rem;
}}
.verdict {{
    border-left: 6px solid {ROSE}; background: #fff0f6; padding: .9rem 1.2rem;
    border-radius: 6px; color: #343a40;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def fetch_pipeline(file_name):
    path = ARTIFACTS / file_name
    if not path.is_file():
        st.error(f"Saved model `{path.name}` is missing. Run `python train_models.py` first.")
        st.stop()
    return joblib.load(path)


@st.cache_data
def fetch_tuned_params():
    try:
        return pd.read_csv(ARTIFACTS / "best_hyperparameters.csv",
                           index_col=0)["Best hyperparameters"].to_dict()
    except Exception:
        return {}


def six_scores(truth, predicted, probability):
    return {
        "Accuracy": accuracy_score(truth, predicted),
        "AUC": roc_auc_score(truth, probability),
        "Precision": precision_score(truth, predicted),
        "Recall": recall_score(truth, predicted),
        "F1": f1_score(truth, predicted),
        "MCC": matthews_corrcoef(truth, predicted),
    }


def read_input(source):
    """Read and validate a CSV; returns (frame, error_message)."""
    try:
        frame = pd.read_csv(source)
    except Exception as exc:
        return None, f"Could not read the CSV: {exc}"
    absent = [c for c in INPUT_COLUMNS + [LABEL] if c not in frame.columns]
    if absent:
        return None, f"Missing required column(s): {', '.join(absent)}"
    if not set(frame[LABEL].dropna().unique()) <= {0, 1}:
        return None, f"`{LABEL}` must contain only 0 (Malignant) and 1 (Benign)."
    if frame[LABEL].nunique() < 2:
        return None, "The CSV has a single class only; AUC and ROC need both Malignant and Benign rows."
    return frame, None


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Tumour Triage Bench</h1>
  <p>Five tuned classifiers, one biopsy dataset. Load a test file, inspect any model, then see who wins.</p>
  <span class="chip">569 biopsies</span><span class="chip">30 nuclear measurements</span>
  <span class="chip">5-fold CV tuning</span><span class="chip">no retraining in-app</span>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — data source
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("1 · Choose test data")
    source_mode = st.radio("Data source", ["Bundled sample (114 rows)",
                                           "Full dataset (569 rows)",
                                           "Upload my own CSV"],
                           label_visibility="collapsed")
    if source_mode == "Full dataset (569 rows)":
        st.warning("455 of these 569 rows were used for training, so scores here are "
                   "optimistic. Use the 114-row sample for honest test results.")
    upload = None
    if source_mode == "Upload my own CSV":
        upload = st.file_uploader("Test CSV", type="csv",
                                  help=f"Needs the {len(INPUT_COLUMNS)} feature columns and `{LABEL}` (0/1).")
    if SAMPLE_CSV.is_file():
        st.download_button("Download the sample CSV", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.divider()
    st.caption("Positive class = **Benign (1)**. All scores are computed on the rows you load, "
               "using models saved by `train_models.py`.")
    st.caption(f"Dataset: [UCI Breast Cancer Wisconsin (Diagnostic)]({UCI_LINK})")

if source_mode == "Upload my own CSV":
    if upload is None:
        st.info("Upload a test CSV in the sidebar, or switch to the bundled sample.")
        st.stop()
    frame, problem = read_input(upload)
elif source_mode == "Full dataset (569 rows)":
    frame, problem = read_input(FULL_DATA_CSV)
else:
    frame, problem = read_input(SAMPLE_CSV)

if frame is None:
    st.error(problem)
    st.stop()

features, truth = frame[INPUT_COLUMNS], frame[LABEL]

# Score every model once; every tab reuses these.
outputs = {}
for model_name, file_name in PIPELINE_FILES.items():
    pipe = fetch_pipeline(file_name)
    hard, soft = pipe.predict(features), pipe.predict_proba(features)[:, 1]
    outputs[model_name] = {"pred": hard, "prob": soft, "scores": six_scores(truth, hard, soft)}

board = pd.DataFrame([{"Model": m, **o["scores"]} for m, o in outputs.items()]).set_index("Model")
board = board.reindex(columns=pd.Index(SCORE_NAMES))

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_data, tab_inspect, tab_board = st.tabs(["Dataset & notes", "Model inspector", "Leaderboard"])

# ---- Tab 1: data ------------------------------------------------------------
with tab_data:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Loaded rows")
        st.dataframe(frame, width="stretch", height=330)
        st.caption(f"{len(frame)} rows · {len(INPUT_COLUMNS)} features")
    with right:
        st.markdown("#### Class mix in this file")
        counts = truth.replace(dict(enumerate(OUTCOMES))).value_counts().reindex(OUTCOMES)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(counts.index, counts.to_numpy(), color=[ROSE, TEAL])
        for y_pos, n in enumerate(counts.to_numpy()):
            ax.text(n, y_pos, f" {n}", va="center")
        ax.set_xlim(0, counts.max() * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.caption(f"Majority-class baseline accuracy: {counts.max() / counts.sum():.3f}")

    with st.expander("How the models were built"):
        st.markdown(
            """
- **Pipeline:** `StandardScaler` → classifier, saved as one file, so raw feature columns go straight in.
- **Tuning:** `GridSearchCV`, 5-fold stratified CV on the 455-row training split, refit on F1 (Benign as positive).
- **Class balance:** about 63% Benign / 37% Malignant. Rows were **not** resampled; `class_weight` / priors
  were in each grid and kept only where CV preferred them.
- **Test split:** 114 held-out rows (20%, stratified, `random_state=42`).
"""
        )

# ---- Tab 2: inspector ---------------------------------------------------------
with tab_inspect:
    chosen = st.selectbox("Model to inspect", list(PIPELINE_FILES), key="chosen_model")
    result = outputs[chosen]
    tuned = fetch_tuned_params().get(chosen)
    if tuned:
        st.caption(f"Tuned settings: `{tuned}`")

    for slot, key in zip(st.columns(6), SCORE_NAMES):
        slot.metric(key, f"{result['scores'][key]:.4f}", border=True)

    cm_col, roc_col = st.columns(2)
    with cm_col:
        st.markdown("##### Confusion matrix")
        grid = confusion_matrix(truth, result["pred"])
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.imshow(grid, cmap=HEAT)
        ax.set_xticks([0, 1], OUTCOMES)
        ax.set_yticks([0, 1], OUTCOMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, grid[r, c], ha="center", va="center", fontsize=16, fontweight="bold",
                        color="white" if grid[r, c] > grid.max() / 2 else SLATE)
        st.pyplot(fig)
        missed = int(grid[0, 1])
        st.caption(f"{missed} malignant biopsies were called Benign (false negatives from the "
                   "clinical point of view).")
    with roc_col:
        st.markdown("##### ROC curve")
        fpr, tpr, _ = roc_curve(truth, result["prob"])
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.fill_between(fpr, tpr, alpha=0.15, color=MODEL_COLOURS[chosen])
        ax.plot(fpr, tpr, color=MODEL_COLOURS[chosen], lw=2.2, label=f"AUC {result['scores']['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], ls=":", color="grey")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    st.markdown("##### Classification report")
    rep = pd.DataFrame(classification_report(truth, result["pred"], target_names=OUTCOMES,
                                             output_dict=True)).T
    rep["support"] = rep["support"].astype(int)
    st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                   "f1-score": "{:.3f}"}), width="stretch")

# ---- Tab 3: leaderboard -------------------------------------------------------
with tab_board:
    ranked = board.sort_values(by=["F1", "MCC"], ascending=[False, False])
    top = ranked.index[0]
    st.markdown(
        f"""<div class="verdict"><b>Best on this file: {top}</b> — F1 {ranked.loc[top, 'F1']:.4f},
        MCC {ranked.loc[top, 'MCC']:.4f}. Order: {' › '.join(ranked.index)} (F1, then MCC as tie-break).</div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    st.dataframe(ranked.round(4).style.background_gradient(cmap="GnBu", axis=0),
                 width="stretch")

    st.markdown("##### One panel per metric")
    fig, panels = plt.subplots(1, 6, figsize=(17, 3.4), sharey=True)
    order = list(ranked.index)
    for ax, metric in zip(panels, SCORE_NAMES):
        vals = board.loc[order, metric]
        ax.barh(order, vals.to_numpy(), color=[MODEL_COLOURS[m] for m in order])
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.set_xlim(max(0.0, vals.min() - 0.1), 1.0)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        for y_pos, v in enumerate(vals.to_numpy()):
            ax.text(v, y_pos, f"{v:.2f} ", ha="right", va="center", color="white", fontsize=8)
    st.pyplot(fig)

    st.markdown("##### All ROC curves together")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in PIPELINE_FILES:
        fpr, tpr, _ = roc_curve(truth, outputs[m]["prob"])
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLOURS[m], label=f"{m} ({board.loc[m, 'AUC']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

st.caption("Models are loaded from `model/` and only score the rows above; nothing is retrained.")
