import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# STYLE
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# PATHS
DATA_PATH = "data/processed/features.csv"

BASE_OUTPUT = "outputs/clustering/"
DATA_OUT = BASE_OUTPUT + "data/"
FIG_PATH = BASE_OUTPUT + "figures/"

os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# LOAD


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)
    return df

# FEATURE ENGINEERING


def engineer_features(df):
    race_max_lap = df.groupby("raceId")["first_stop_lap"].transform("max")
    df["race_length_proxy"] = race_max_lap + 5
    df["pit_timing_ratio"] = df["first_stop_lap"] / df["race_length_proxy"]
    df["avg_time_per_stop"] = df["total_stop_time"] / \
        df["num_stops"].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# SCALE


def scale_features(df):
    features = [
        "num_stops",
        "pit_timing_ratio",
        "avg_time_per_stop",
        "position_change",
        "grid"
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, features

# DBSCAN


def run_dbscan(df, X):
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)

    distances = np.sort(distances[:, 4])
    eps = np.percentile(distances, 92)

    plt.figure(figsize=(7, 4))
    plt.plot(distances)
    plt.axhline(eps, linestyle="--")
    plt.title(f"DBSCAN k-distance (eps ≈ {eps:.2f})")
    plt.xlabel("Points (sorted)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "dbscan_kdist.png")
    plt.close()

    db = DBSCAN(eps=eps, min_samples=5)
    df["dbscan_label"] = db.fit_predict(X)

    return df

# K SELECTION


def find_best_k(X):
    scores = []
    K_range = range(2, 8)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = kmeans.fit_predict(X)
        scores.append(silhouette_score(X, labels))

    best_k = K_range[np.argmax(scores)]

    plt.figure(figsize=(7, 4))
    plt.plot(K_range, scores, marker='o')
    plt.axvline(best_k, linestyle="--")
    plt.title("Silhouette Score by K")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "kmeans_k_selection.png")
    plt.close()

    return best_k

# KMEANS


def run_kmeans(df, X):
    mask = df["dbscan_label"] != -1
    X_clean = X[mask]

    best_k = find_best_k(X_clean)

    kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)

    df["cluster"] = -1
    df.loc[mask, "cluster"] = kmeans.fit_predict(X_clean)

    score = silhouette_score(X_clean, df.loc[mask, "cluster"])
    print(f"\nSilhouette Score: {score:.3f}")

    with open(DATA_OUT + "metrics.json", "w") as f:
        json.dump({"silhouette_score": float(score)}, f)

    return df, kmeans

# STRATEGY LABELING


def assign_strategy_names(df):
    summary = df[df["cluster"] != -
                 1].groupby("cluster").mean(numeric_only=True)

    summary["stop_rank"] = summary["num_stops"].rank(
        ascending=False, method="dense")
    summary["early_rank"] = summary["pit_timing_ratio"].rank(method="dense")
    summary["late_rank"] = summary["pit_timing_ratio"].rank(
        ascending=False, method="dense")

    strategy_map = {}

    for cid, row in summary.iterrows():
        if row["stop_rank"] == 1:
            strategy = "Aggressive Multi-Stop"
        elif row["early_rank"] == 1:
            strategy = "Early Pit Strategy"
        elif row["late_rank"] == 1:
            strategy = "Late Pit Strategy"
        else:
            strategy = "Standard Strategy"

        strategy_map[cid] = strategy

    df["strategy"] = df["cluster"].map(strategy_map)

    df.loc[(df["cluster"] == -1) & (df["dnf"] == 1),
           "strategy"] = "Outlier (DNF)"
    df.loc[(df["cluster"] == -1) & (df["dnf"] == 0),
           "strategy"] = "Outlier (Anomalous)"

    return df, strategy_map

# VISUALS


def create_visuals(df, X, kmeans):
    mask = df["cluster"] != -1
    df_clean = df[mask]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X[mask])
    centers = pca.transform(kmeans.cluster_centers_)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df_clean["strategy"],
        palette="colorblind",
        alpha=0.6,
        s=35
    )

    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=150, marker="X")

    plt.title("F1 Strategy Clusters (PCA Projection)")
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(FIG_PATH + "cluster_plot.png")
    plt.close()

    summary = df_clean.groupby("strategy")[
        ["num_stops", "pit_timing_ratio",
            "avg_time_per_stop", "position_change", "grid"]
    ].mean()

    plt.figure(figsize=(9, 5))
    sns.heatmap(summary, annot=True, fmt=".2f", cmap="vlag")
    plt.title("Strategy Feature Averages")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "heatmap.png")
    plt.close()

# RUN


def run():
    df = load_data()
    df = engineer_features(df)
    X, _ = scale_features(df)
    df = run_dbscan(df, X)
    df, kmeans = run_kmeans(df, X)
    df, strategy_map = assign_strategy_names(df)
    create_visuals(df, X, kmeans)

    df.to_csv(DATA_OUT + "strategy_clusters.csv", index=False)

    with open(DATA_OUT + "strategy_map.json", "w") as f:
        json.dump(strategy_map, f, indent=4)

    print("\nDONE")


if __name__ == "__main__":
    run()
