import pandas as pd
import numpy as np
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
import joblib
import os
import time

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
#  MODEL REGISTRY
# ══════════════════════════════════════════════════════
UNSUPERVISED_MODELS = {
    "Isolation Forest"    : "Detects anomalies by isolating data points. Best for high-dimensional sensor data.",
    "Local Outlier Factor": "Detects anomalies based on local density. Great for clustered data.",
    "One-Class SVM"       : "Learns a boundary around normal data. Good for small/clean datasets.",
}

SUPERVISED_MODELS = {
    "Random Forest"      : "Ensemble of decision trees. Most accurate when labels are available.",
    "Decision Tree"      : "Simple tree model. Easy to interpret and explain.",
    "Gradient Boosting"  : "Boosted trees — high accuracy, slower to train.",
    "K-Nearest Neighbors": "Classifies based on similarity to nearest neighbors.",
    "Logistic Regression": "Fast linear baseline model.",
    "Naive Bayes"        : "Probabilistic model. Extremely fast on large datasets.",
}

ALL_MODELS = {**UNSUPERVISED_MODELS, **SUPERVISED_MODELS}


# ══════════════════════════════════════════════════════
#  AUTO CONTAMINATION
# ══════════════════════════════════════════════════════
def auto_contamination(y_train=None, contamination_hint=0.1):
    """
    If true labels are available, calculate the exact anomaly ratio.
    Otherwise use the user-provided hint.
    Clamps result between 0.01 and 0.49 as required by sklearn.
    """
    if y_train is not None:
        ratio = float(np.mean(y_train))
        ratio = max(0.01, min(0.49, ratio))
        return round(ratio, 4)
    return contamination_hint


# ══════════════════════════════════════════════════════
#  BEST THRESHOLD FINDER
# ══════════════════════════════════════════════════════
def find_best_threshold(scores: np.ndarray, y_true: np.ndarray):
    """
    Scan thresholds from 1% to 50% and find the one that
    gives the highest F1 score.

    Returns:
        best_threshold_pct : int (e.g. 15)
        best_f1            : float
        all_f1s            : list of F1 for each threshold (for plotting)
        all_precs          : list of Precision for each threshold
        all_recs           : list of Recall for each threshold
    """
    thresholds = list(range(1, 51))
    f1s, precs, recs = [], [], []

    for t in thresholds:
        preds = predict_with_threshold(scores, t)
        f1s.append(f1_score(y_true, preds, zero_division=0))
        precs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds, zero_division=0))

    best_idx = int(np.argmax(f1s))
    return (
        thresholds[best_idx],
        round(f1s[best_idx], 4),
        f1s, precs, recs
    )


# ══════════════════════════════════════════════════════
#  ISOLATION FOREST AUTO-TUNER
# ══════════════════════════════════════════════════════
def tune_isolation_forest(X_train, X_test, y_true=None, contamination=0.1):
    """
    Try multiple Isolation Forest configurations and return the best one
    based on F1 score (if labels available) or anomaly score spread.

    Configurations tested:
        - Different n_estimators: 100, 200, 300
        - Different max_samples: 'auto', 256, 512
        - Different max_features: 1.0, 0.8

    Returns:
        best_model      : trained IsolationForest
        best_preds      : predictions from best model
        best_scores     : anomaly scores from best model
        best_config     : dict of best hyperparameters
        best_f1         : float
        all_configs     : list of all tried configs with their scores
    """
    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    X_te = X_test.values  if hasattr(X_test,  "values") else X_test

    configs = [
        {"n_estimators": 100, "max_samples": "auto", "max_features": 1.0},
        {"n_estimators": 200, "max_samples": "auto", "max_features": 1.0},
        {"n_estimators": 300, "max_samples": "auto", "max_features": 1.0},
        {"n_estimators": 200, "max_samples": 256,    "max_features": 1.0},
        {"n_estimators": 200, "max_samples": 512,    "max_features": 1.0},
        {"n_estimators": 200, "max_samples": "auto", "max_features": 0.8},
        {"n_estimators": 300, "max_samples": 512,    "max_features": 0.8},
    ]

    best_model  = None
    best_preds  = None
    best_scores = None
    best_config = None
    best_f1     = -1
    all_configs = []

    for cfg in configs:
        m = IsolationForest(
            n_estimators  = cfg["n_estimators"],
            max_samples   = cfg["max_samples"],
            max_features  = cfg["max_features"],
            contamination = contamination,
            random_state  = 42,
            n_jobs        = -1
        )
        m.fit(X_tr)
        raw_preds = m.predict(X_te)
        preds     = np.where(raw_preds == -1, 1, 0)
        scores    = -m.decision_function(X_te)

        if y_true is not None:
            # Use best threshold F1 for fair comparison
            _, f1, _, _, _ = find_best_threshold(scores, y_true[:len(scores)])
        else:
            # Without labels: use score spread as quality proxy
            f1 = float(np.std(scores))

        all_configs.append({**cfg, "f1": round(f1, 4)})

        if f1 > best_f1:
            best_f1     = f1
            best_model  = m
            best_preds  = preds
            best_scores = scores
            best_config = cfg

    safe = "isolation_forest"
    joblib.dump(best_model, f"{MODEL_DIR}/{safe}.pkl")

    return best_model, best_preds, best_scores, best_config, round(best_f1, 4), all_configs


# ══════════════════════════════════════════════════════
#  THRESHOLD PREDICTION
# ══════════════════════════════════════════════════════
def predict_with_threshold(scores: np.ndarray, threshold_percentile: float) -> np.ndarray:
    """
    Mark top X% of anomaly scores as anomalies.

    Parameters:
        scores               : anomaly score array (higher = more anomalous)
        threshold_percentile : float between 1 and 50

    Returns:
        predictions : array of 0 (normal) or 1 (anomaly)
    """
    cutoff = np.percentile(scores, 100 - threshold_percentile)
    return (scores >= cutoff).astype(int)


# ══════════════════════════════════════════════════════
#  MAIN RUN MODEL
# ══════════════════════════════════════════════════════
def run_model(name, X_train, X_test, y_train=None, contamination=0.1,
              auto_tune=False, y_true_for_tuning=None):
    """
    Train and predict using specified model.

    Parameters:
        name                : model name
        X_train             : training features
        X_test              : test features
        y_train             : labels for supervised models
        contamination       : anomaly rate
        auto_tune           : if True, runs Isolation Forest tuner
        y_true_for_tuning   : true labels used for tuning (not training)

    Returns:
        predictions    : array of 0/1
        scores         : anomaly scores
        train_time     : seconds
        extra_info     : dict with tuning results (if auto_tune=True)
    """
    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    X_te = X_test.values  if hasattr(X_test,  "values") else X_test
    extra_info = {}

    start = time.time()

    if name == "Isolation Forest":
        if auto_tune:
            _, preds, scores, best_cfg, best_f1, all_cfgs = tune_isolation_forest(
                X_tr, X_te,
                y_true=y_true_for_tuning,
                contamination=contamination
            )
            extra_info = {
                "tuned"      : True,
                "best_config": best_cfg,
                "best_f1"    : best_f1,
                "all_configs": all_cfgs
            }
        else:
            m = IsolationForest(
                n_estimators=200, contamination=contamination,
                random_state=42, n_jobs=-1
            )
            m.fit(X_tr)
            preds  = np.where(m.predict(X_te) == -1, 1, 0)
            scores = -m.decision_function(X_te)
            joblib.dump(m, f"{MODEL_DIR}/isolation_forest.pkl")

    elif name == "Local Outlier Factor":
        m = LocalOutlierFactor(
            n_neighbors=20, contamination=contamination,
            novelty=True, n_jobs=-1
        )
        m.fit(X_tr)
        preds  = np.where(m.predict(X_te) == -1, 1, 0)
        scores = -m.decision_function(X_te)
        joblib.dump(m, f"{MODEL_DIR}/local_outlier_factor.pkl")

    elif name == "One-Class SVM":
        max_s = min(5000, len(X_tr))
        idx   = np.random.choice(len(X_tr), max_s, replace=False)
        m = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")
        m.fit(X_tr[idx])
        preds  = np.where(m.predict(X_te) == -1, 1, 0)
        scores = -m.decision_function(X_te)
        joblib.dump(m, f"{MODEL_DIR}/one_class_svm.pkl")

    elif name == "Random Forest":
        m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]
        joblib.dump(m, f"{MODEL_DIR}/random_forest.pkl")

    elif name == "Decision Tree":
        m = DecisionTreeClassifier(random_state=42, max_depth=10)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]
        joblib.dump(m, f"{MODEL_DIR}/decision_tree.pkl")

    elif name == "Gradient Boosting":
        m = GradientBoostingClassifier(n_estimators=100, random_state=42)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]
        joblib.dump(m, f"{MODEL_DIR}/gradient_boosting.pkl")

    elif name == "K-Nearest Neighbors":
        m = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]
        joblib.dump(m, f"{MODEL_DIR}/k-nearest_neighbors.pkl")

    elif name == "Logistic Regression":
        m = LogisticRegression(random_state=42, max_iter=500, n_jobs=-1)
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]
        joblib.dump(m, f"{MODEL_DIR}/logistic_regression.pkl")

    elif name == "Naive Bayes":
        m = GaussianNB()
        m.fit(X_tr, y_train)
        preds  = m.predict(X_te)
        scores = m.predict_proba(X_te)[:, 1]
        joblib.dump(m, f"{MODEL_DIR}/naive_bayes.pkl")

    else:
        raise ValueError(f"Unknown model: {name}")

    train_time = round(time.time() - start, 2)
    return preds.astype(int), scores, train_time, extra_info


# ══════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════
def evaluate(y_true, y_pred, model_name="", train_time=0):
    """Return dict of evaluation metrics."""
    return {
        "model"            : model_name,
        "accuracy"         : round(accuracy_score(y_true, y_pred), 4),
        "precision"        : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"           : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score"         : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix" : confusion_matrix(y_true, y_pred).tolist(),
        "report"           : classification_report(
            y_true, y_pred, target_names=["Normal", "Anomaly"]
        ),
        "train_time"       : train_time,
    }