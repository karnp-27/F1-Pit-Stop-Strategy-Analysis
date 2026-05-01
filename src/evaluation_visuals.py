import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# STYLE
sns.set_theme(style="whitegrid", context="talk")

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# PATHS
BASE_OUTPUT = "outputs/final_results/"
DATA_OUT = BASE_OUTPUT + "data/"
FIG_PATH = BASE_OUTPUT + "figures/"

CLUSTER_PATH = "outputs/clustering/data/strategy_clusters.csv"
CLASS_PATH = "outputs/classification/data/strategy_classification.csv"

os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# LOAD DATA


def load_data():

    clusters = pd.read_csv(CLUSTER_PATH)
    classification = pd.read_csv(CLASS_PATH)

    print("Clusters:", clusters.shape)
    print("Classification:", classification.shape)

    return clusters, classification

# SAFE MERGE (FIXED)


def merge_data(clusters, classification):

    df = clusters.merge(
        classification,
        on=["raceId", "driverId"],
        how="inner",
        suffixes=("_cluster", "_class")
    )

    print("Merged dataset:", df.shape)

    df.to_csv(DATA_OUT + "strategy_evaluation.csv", index=False)

    return df

# SAFE COLUMN RESOLVER (IMPORTANT FIX)


def get_col(df, base_name):
    """
    Safely finds column even if merge added suffixes.
    """

    if base_name in df.columns:
        return base_name

    if base_name + "_cluster" in df.columns:
        return base_name + "_cluster"

    if base_name + "_class" in df.columns:
        return base_name + "_class"

    raise KeyError(f"Column '{base_name}' not found in dataframe")

# 1. STRATEGY SUCCESS RATE


def plot_strategy_success(df):

    summary = df.groupby("strategy")["actual"].mean().sort_values()

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=summary.values,
        y=summary.index
    )

    plt.title("Strategy Success Rate (Probability of Position Improvement)")
    plt.xlabel("Success Rate")
    plt.ylabel("Strategy")

    plt.tight_layout()
    plt.savefig(FIG_PATH + "strategy_success_rate.png")
    plt.close()

# 2. CLUSTER PERFORMANCE (FIXED)


def plot_cluster_performance(df):

    col = get_col(df, "position_change")

    summary = df.groupby("strategy")[col].mean().sort_values()

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=summary.values,
        y=summary.index
    )

    plt.title("Average Position Change by Strategy")
    plt.xlabel("Avg Position Change")
    plt.ylabel("Strategy")

    plt.tight_layout()
    plt.savefig(FIG_PATH + "cluster_performance.png")
    plt.close()

# 3. MODEL ACCURACY BY STRATEGY


def plot_model_accuracy_by_strategy(df):

    df["log_correct"] = (df["actual"] == df["log_pred"]).astype(int)
    df["svm_correct"] = (df["actual"] == df["svm_pred"]).astype(int)

    summary = df.groupby("strategy")[["log_correct", "svm_correct"]].mean()

    summary.plot(kind="bar", figsize=(10, 5))

    plt.title("Model Accuracy by Strategy Type")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(FIG_PATH + "model_accuracy_by_strategy.png")
    plt.close()

# 4. STRATEGY DISTRIBUTION


def plot_strategy_distribution(df):

    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df,
        y="strategy",
        order=df["strategy"].value_counts().index
    )

    plt.title("Strategy Distribution")
    plt.xlabel("Count")
    plt.ylabel("Strategy")

    plt.tight_layout()
    plt.savefig(FIG_PATH + "strategy_distribution.png")
    plt.close()

# SUMMARY TABLE


def create_summary_table(df):

    col = get_col(df, "position_change")

    summary = df.groupby("strategy").agg({
        "actual": "mean",
        col: "mean",
        "log_prob": "mean",
        "svm_prob": "mean"
    }).rename(columns={
        "actual": "success_rate",
        col: "avg_position_change"
    })

    summary.to_csv(DATA_OUT + "strategy_summary_table.csv")

    print("\nSaved summary table:")
    print(summary)

# RUN PIPELINE


def run():

    clusters, classification = load_data()

    df = merge_data(clusters, classification)

    # VISUALS
    plot_strategy_success(df)
    plot_cluster_performance(df)
    plot_model_accuracy_by_strategy(df)
    plot_strategy_distribution(df)

    # SUMMARY
    create_summary_table(df)

    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    run()
