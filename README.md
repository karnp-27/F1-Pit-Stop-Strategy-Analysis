# Description
This project builds a machine learning pipeline to analyze Formula 1 pit stop strategies and their impact on race outcomes. 
It combines clustering and classification techniques to identify strategy patterns and evaluate how pit behavior influences performance.

## Overview

The pipeline performs the following key tasks:
- Uses clustering techniques to group similar pit stop behaviors into interpretable strategy types  
- Applies classification models to predict whether a driver improves their race position  
- Assesses the relationship between pit stop strategies and race outcomes  

## Installation
Step 1 — Clone or Download Repository
GitHub Link: https://github.com/karnp-27/F1-Pit-Stop-Strategy-Analysis

## Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn

## Data
All datasets are included locally within the repository. 

Directory: `data/raw/`

No external data downloads are required. 
The dataset was sourced from Kaggle: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?select=pit_stops.csv

## Execution
Run full pipeline from run_pipeline.py
How It Works: 
The run_pipeline.py script orchestrates the full workflow by sequentially executing all modules inside the src/ directory. 
Each module performs a specific stage of the analysis and writes outputs to disk.

**Step 1 - Data Preparationn** (data_cleaning.py)
Install required dependencies using:
What it does:
- Loads raw datasets
- Cleans missing or invalid values
- Joins datasets into a unified structure
- Engineers race-level and driver-level features

Output:
- data/processed/features.csv

**Step 2 - Strategy Discovery** (clustering.py)
What it does:
- Scales features for clustering
- Uses DBSCAN to detect and remove outliers
- Applies K-Means clustering to group pit stop strategies
- Uses silhouette score to select the optimal number of clusters
- Assigns interpretable strategy labels

Strategy labels include:
- Aggressive Multi-Stop
- Early Pit Strategy
- Late Pit Strategy
- Standard Strategy
- Outliers (DNF / anomalous cases)

Outputs:
- outputs/clustering/data/strategy_clusters.csv
- outputs/clustering/data/strategy_map.json
- outputs/clustering/data/metrics.json

Figures:
- outputs/clustering/figures/

**Step 3 - Outcome Prediction** (classification.py)
What it does:
- Builds models to predict whether a driver improves race position
- Uses pit stop features as inputs

Models used:
- Logistic Regression
- Support Vector Machine (SVM)

Evaluation metrics:
- Accuracy
- F1 Score
- ROC-AUC
- Cross-validation F1

Outputs:
- outputs/classification/data/strategy_classification.csv
- outputs/classification/data/classification_results.json

Figures:
- outputs/classification/figures/

**Step 4 - Strategy Effectiveness** (evaluation_visuals.py)
What it does:
- Merges clustering results with classification predictions
- Compares performance across discovered strategy types
- Analyzes how strategy choice relates to race outcomes

Outputs:
- outputs/final_results/data/strategy_evaluation.csv
- outputs/final_results/data/strategy_summary_table.csv

Figures:
- outputs/final_results/figures/

**Project Structure**
project_root/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── features.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── clustering.py
│   ├── classification.py
│   └── evaluation_visuals.py
│
├── outputs/
│   ├── clustering/
│   ├── classification/
│   └── final_results/
│
├── run_pipeline.py
└── README.md

## Key Results
- Aggressive multi-stop strategies are associated with higher success rates  
- Early pit strategies tend to underperform, often indicating reactive decisions  
- Outliers (DNFs) show consistently poor outcomes, validating the pipeline  
- Clustering produced weak separation (silhouette ~0.21), suggesting strategies exist on a continuum rather than in clearly defined groups  
