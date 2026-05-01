import sys
import os
import traceback
from datetime import datetime

# Ensure src/ is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# IMPORT PIPELINE MODULES
try:
    import data_cleaning
    import clustering
    import classification
    import evaluation_visuals
except ImportError as e:
    print("Import error:", e)
    print("Make sure src/ folder exists and contains all scripts.")
    sys.exit(1)

# LOGGER


def log_step(step_name):
    print("\n" + "="*60)
    print(f"{step_name}")
    print("="*60)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# SAFE EXECUTION WRAPPER


def run_step(func, name):
    try:
        log_step(name)
        func()
        print(f"Completed: {name}")
    except Exception as e:
        print(f"Error in {name}")
        traceback.print_exc()
        sys.exit(1)

# MAIN PIPELINE


def main():

    print("\n" + "#"*70)
    print("F1 PIT STOP STRATEGY ANALYSIS PIPELINE")
    print("#"*70)

    # Step 1: Data Cleaning
    run_step(
        data_cleaning.clean_and_merge,
        "Step 1: Data Cleaning & Feature Generation"
    )

    # Step 2: Clustering
    run_step(
        clustering.run,
        "Step 2: Strategy Clustering"
    )

    # Step 3: Classification
    run_step(
        classification.run,
        "Step 3: Outcome Prediction (Classification)"
    )

    # Step 4: Evaluation & Visualization
    run_step(
        evaluation_visuals.run,
        "Step 4: Strategy Evaluation & Insights"
    )

    print("\n" + "#"*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("#"*70)

    print("\nOutputs generated in:")
    print("data/processed/")
    print("outputs/")
    print("figures in respective subfolders\n")


# ENTRY POINT
if __name__ == "__main__":
    main()
