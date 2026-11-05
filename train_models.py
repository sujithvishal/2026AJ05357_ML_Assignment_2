"""
train_models.py — standalone version of ml2_assignment.ipynb (Sections 3-7).

Loads the Breast Cancer Wisconsin (Diagnostic) data, creates the stratified 80/20 split,
tunes all five classifiers with 5-fold stratified GridSearchCV (refit on F1),
evaluates them on the held-out test set and saves the pipelines to model/.

Class imbalance: the data is NOT resampled. Class weighting is a tuned
hyperparameter, chosen per model by cross-validation.

Usage:
    python train_models.py
"""
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RAW_PATH = "data/breast_cancer_wisconsin_full.csv"
MODEL_DIR = "model"
TARGET_COL = "target"          # 0 = Malignant, 1 = Benign

FILE_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return df, feature_cols


def make_split(df, feature_cols):
    X, y = df[feature_cols], df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def model_grids():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10, 100],
             "model__class_weight": [None, "balanced"]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [3, 4, 5, 6, 8, None],
             "model__min_samples_leaf": [1, 5, 10, 20],
             "model__criterion": ["gini", "entropy"],
             "model__class_weight": [None, "balanced"]}),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7, 11, 15, 21],
             "model__weights": ["uniform", "distance"],
             "model__p": [1, 2]}),
        "Naive Bayes": (
            GaussianNB(),
            {"model__var_smoothing": np.logspace(-9, 0, 10),
             "model__priors": [None, [0.5, 0.5]]}),
        "Support Vector Machine": (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1, 10],
             "model__gamma": ["scale", 0.001, 0.01],
             "model__class_weight": [None, "balanced"]}),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df, feature_cols = load_data()
    X_train, X_test, y_train, y_test = make_split(df, feature_cols)
    print(f"Rows: {len(df)} | features {len(feature_cols)} | train {len(X_train)} | "
          f"test {len(X_test)} | benign rate {y_train.mean():.3f}")

    test_data = X_test.copy()
    test_data[TARGET_COL] = y_test.values
    test_data.to_csv("test_data.csv", index=False)

    # Scaling lives inside every pipeline so each saved model accepts raw features
    # and is refitted per CV fold (no leakage).
    preprocessor = ColumnTransformer([("num", StandardScaler(), feature_cols)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": "f1", "mcc": make_scorer(matthews_corrcoef), "auc": "roc_auc",
               "accuracy": "accuracy", "precision": "precision", "recall": "recall"}

    os.makedirs(MODEL_DIR, exist_ok=True)
    results, params, balance_rows = {}, {}, []
    for name, (clf, grid) in model_grids().items():
        search = GridSearchCV(Pipeline([("preprocess", preprocessor), ("model", clf)]),
                              grid, scoring=scoring, refit="f1", cv=cv, n_jobs=-1)
        t0 = time.time()
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        metrics = evaluate(y_test, pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1])
        metrics.update({"CV F1": search.best_score_,
                        "Tuning Time (s)": round(time.time() - t0, 1),
                        "Refit Time (s)": round(search.refit_time_, 3)})
        results[name] = metrics
        params[name] = ", ".join(f"{k.replace('model__', '')}={v}"
                                 for k, v in search.best_params_.items())
        joblib.dump(pipe, f"{MODEL_DIR}/{FILE_NAMES[name]}.joblib", compress=3)

        # Balanced vs natural weighting, best CV config of each (training data only)
        cvr = pd.DataFrame(search.cv_results_)
        col = next((c for c in cvr.columns
                    if c in ("param_model__class_weight", "param_model__priors")), None)
        if col is not None:
            mode = cvr[col].apply(lambda v: "balanced" if isinstance(v, (str, list))
                                  else "none (natural ratio)")
            for m, sub in cvr.groupby(mode):
                best = sub.loc[sub["mean_test_f1"].idxmax()]
                balance_rows.append({
                    "Model": name, "Weighting": m,
                    **{k: best[f"mean_test_{s}"] for k, s in
                       [("CV Accuracy", "accuracy"), ("CV AUC", "auc"),
                        ("CV Precision", "precision"), ("CV Recall", "recall"),
                        ("CV F1", "f1"), ("CV MCC", "mcc")]}})
        print(f"✓ {name:<24} test F1 {metrics['F1']:.4f} | {params[name]}")

    results_df = pd.DataFrame(results).T
    results_df.iloc[:, :7] = results_df.iloc[:, :7].astype(float).round(4)
    results_df.to_csv(f"{MODEL_DIR}/comparison_table.csv")
    pd.DataFrame({"Best hyperparameters": params}).to_csv(f"{MODEL_DIR}/best_hyperparameters.csv")
    pd.DataFrame(balance_rows).set_index(["Model", "Weighting"]).round(4).to_csv(
        f"{MODEL_DIR}/class_balance_cv.csv")
    print("\n", results_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].to_string())
    print(f"\nWinner by F1: {results_df['F1'].astype(float).idxmax()}")


if __name__ == "__main__":
    main()
