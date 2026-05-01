import pandas as pd
import os

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

os.makedirs(PROCESSED_PATH, exist_ok=True)


def load_data():
    pit = pd.read_csv(RAW_PATH + "pit_stops.csv")
    results = pd.read_csv(RAW_PATH + "results.csv")
    races = pd.read_csv(RAW_PATH + "races.csv")
    return pit, results, races


def clean_and_merge():
    pit, results, races = load_data()

    # CLEAN PIT STOPS
    pit["duration"] = pd.to_numeric(pit["duration"], errors="coerce")
    pit = pit.dropna(subset=["duration", "lap"])

    pit_agg = pit.groupby(["raceId", "driverId"]).agg(
        num_stops=("lap", "count"),
        first_stop_lap=("lap", "min"),
        avg_stop_time=("duration", "mean"),
        total_stop_time=("duration", "sum")
    ).reset_index()

    # CLEAN RESULTS
    results = results[[
        "raceId", "driverId", "grid", "positionOrder", "statusId"
    ]].copy()

    results = results[(results["grid"].notna()) & (results["grid"] > 0)]
    results = results.dropna(subset=["positionOrder"])

    results["dnf"] = (results["statusId"] != 1).astype(int)

    # FILTER VALID RACES
    valid_races = pit_agg["raceId"].unique()
    results = results[results["raceId"].isin(valid_races)]

    # MERGE
    df = results.merge(pit_agg, on=["raceId", "driverId"], how="left")

    df = df.merge(
        races[["raceId", "year"]],
        on="raceId",
        how="left"
    )

    # HANDLE MISSING
    df["num_stops"] = df["num_stops"].fillna(0)
    df["first_stop_lap"] = df["first_stop_lap"].fillna(
        df["first_stop_lap"].median())
    df["avg_stop_time"] = df["avg_stop_time"].fillna(
        df["avg_stop_time"].median())
    df["total_stop_time"] = df["total_stop_time"].fillna(0)

    # Keep only meaningful strategies
    df = df[df["num_stops"] > 0]

    # FEATURES
    df["position_improved"] = (df["positionOrder"] < df["grid"]).astype(int)
    df["position_change"] = df["grid"] - df["positionOrder"]

    df = df.rename(columns={"positionOrder": "finish"})

    # SAVE
    print("\nSanity check:")
    print(df.describe())

    df.to_csv(PROCESSED_PATH + "features.csv", index=False)
    print("\nCleaned dataset saved!")


if __name__ == "__main__":
    clean_and_merge()
