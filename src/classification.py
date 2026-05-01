import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# -----------------------------
# STYLE
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# -----------------------------
# PATHS (UPDATED)
# -----------------------------
DATA_PATH = "data/processed/features.csv"

BASE_OUTPUT = "outputs/classification/"
FIG_PATH = BASE_OUTPUT + "figures/"
DATA_OUTPUT_PATH = BASE_OUTPUT + "data/"

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(DATA_OUTPUT_PATH, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)
    return df

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------


def engineer_features(df):

    race_max_lap = df.groupby("raceId")["first_stop_lap"].transform("max")
    df["race_length_proxy"] = race_max_lap + 5

    df["pit_timing_ratio"] = df["first_stop_lap"] / df["race_length_proxy"]

    df["avg_time_per_stop"] = df["total_stop_time"] / \
        df["num_stops"].replace(0, np.nan)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df

# -----------------------------
# FEATURES
# -----------------------------


def get_features():

    return [
        "num_stops",
        "pit_timing_ratio",
        "avg_time_per_stop",
        "total_stop_time",
        "grid"
    ]

# -----------------------------
# PREPARE DATA
# -----------------------------


def prepare_data(df, features):

    X = df[features]
    y = df["position_improved"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# -----------------------------
# TRAIN MODELS
# -----------------------------


def train_models(X_train, X_test, y_train, y_test):

    results = {}

    # -----------------------------
    # LOGISTIC REGRESSION
    # -----------------------------
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    log_probs = log_model.predict_proba(X_test)[:, 1]
    log_preds = (log_probs > 0.5).astype(int)

    log_cv = cross_val_score(log_model, X_train, y_train, cv=5, scoring="f1")

    results["logistic"] = {
        "accuracy": accuracy_score(y_test, log_preds),
        "f1": f1_score(y_test, log_preds),
        "roc_auc": roc_auc_score(y_test, log_probs),
        "cv_f1_mean": log_cv.mean()
    }

    # -----------------------------
    # SVM
    # -----------------------------
    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)

    svm_probs = svm_model.predict_proba(X_test)[:, 1]
    svm_preds = (svm_probs > 0.5).astype(int)

    svm_cv = cross_val_score(svm_model, X_train, y_train, cv=5, scoring="f1")

    results["svm"] = {
        "accuracy": accuracy_score(y_test, svm_preds),
        "f1": f1_score(y_test, svm_preds),
        "roc_auc": roc_auc_score(y_test, svm_probs),
        "cv_f1_mean": svm_cv.mean()
    }

    return results, log_model, svm_model, log_probs, svm_probs, log_preds, svm_preds

# -----------------------------
# FINAL CSV OUTPUT
# -----------------------------


def save_strategy_predictions(df, X_scaled, y, log_model, svm_model, features):

    log_probs = log_model.predict_proba(X_scaled)[:, 1]
    svm_probs = svm_model.predict_proba(X_scaled)[:, 1]

    log_preds = (log_probs > 0.5).astype(int)
    svm_preds = (svm_probs > 0.5).astype(int)

    out = df.copy()

    out["actual"] = y.values
    out["log_prob"] = log_probs
    out["log_pred"] = log_preds
    out["svm_prob"] = svm_probs
    out["svm_pred"] = svm_preds
    out["model_agreement"] = (log_preds == svm_preds).astype(int)

    out.to_csv(DATA_OUTPUT_PATH + "strategy_classification.csv", index=False)

    print("Saved strategy_classification.csv:", out.shape)

# -----------------------------
# PLOTS
# -----------------------------


def plot_roc(y_test, log_probs, svm_probs):

    fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr_log, tpr_log, linewidth=2, label="Logistic Regression")
    plt.plot(fpr_svm, tpr_svm, linewidth=2, label="SVM")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.title("ROC Curve: Pit Stop Strategy Classification")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH + "roc_curve.png")
    plt.close()


def plot_confusion(y_test, preds, name):

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Improvement", "Improved"],
        yticklabels=["No Improvement", "Improved"]
    )

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(FIG_PATH + f"{name.lower()}_confusion.png")
    plt.close()


def plot_coefficients(model, feature_names):

    coefs = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs
    }).sort_values("coefficient")

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=coef_df,
        x="coefficient",
        y="feature"
    )

    plt.title("Logistic Regression Feature Effects")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(FIG_PATH + "logistic_coefficients.png")
    plt.close()

# -----------------------------
# SAVE RESULTS
# -----------------------------


def save_outputs(results):

    with open(DATA_OUTPUT_PATH + "classification_results.json", "w") as f:
        json.dump(results, f, indent=4)

# -----------------------------
# MAIN PIPELINE
# -----------------------------


def run():

    df = load_data()
    df = engineer_features(df)

    features = get_features()

    X_train, X_test, y_train, y_test = prepare_data(df, features)

    results, log_model, svm_model, log_probs, svm_probs, log_preds, svm_preds = train_models(
        X_train, X_test, y_train, y_test
    )

    # -----------------------------
    # VISUALS
    # -----------------------------
    plot_roc(y_test, log_probs, svm_probs)
    plot_confusion(y_test, log_preds, "Logistic")
    plot_confusion(y_test, svm_preds, "SVM")
    plot_coefficients(log_model, features)

    # -----------------------------
    # RETRAIN FULL MODEL FOR EXPORT
    # -----------------------------
    X_full = df[features]
    y_full = df["position_improved"]

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    log_model.fit(X_full_scaled, y_full)
    svm_model.fit(X_full_scaled, y_full)

    # -----------------------------
    # SAVE OUTPUTS
    # -----------------------------
    save_outputs(results)

    save_strategy_predictions(
        df,
        X_full_scaled,
        y_full,
        log_model,
        svm_model,
        features
    )

    print("\nDONE")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    run()
