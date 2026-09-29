import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import os


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "data/ml_window_features.csv"

OUTPUT_FOLDER = "data/evaluation"

RESULT_FILE = "data/evaluation/model_metrics.csv"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD ML RESULTS
# ============================================================

print("==========================================")
print("MODEL EVALUATION")
print("==========================================")

print("\nLoading ML window results...")

df = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Total windows: {len(df)}"
)


# ============================================================
# CREATE TRUE LABELS
# ============================================================

# For this simulation:
#
# Normal = 0
# All other conditions = 1


df["Actual"] = df[
    "Condition"
].apply(

    lambda x:
        0 if x == "Normal"
        else 1

)


# ============================================================
# CREATE PREDICTED LABELS
# ============================================================

# ML Status:
#
# Normal  = 0
# Anomaly = 1


df["Predicted"] = df[
    "ML_Status"
].apply(

    lambda x:
        0 if x == "Normal"
        else 1

)


# ============================================================
# CALCULATE METRICS
# ============================================================

actual = df[
    "Actual"
].values


predicted = df[
    "Predicted"
].values


accuracy = accuracy_score(
    actual,
    predicted
)


precision = precision_score(
    actual,
    predicted,
    zero_division=0
)


recall = recall_score(
    actual,
    predicted,
    zero_division=0
)


f1 = f1_score(
    actual,
    predicted,
    zero_division=0
)


# ============================================================
# DISPLAY METRICS
# ============================================================

print("\n==========================================")

print("OVERALL MODEL PERFORMANCE")

print("==========================================")


print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")

print("CLASSIFICATION REPORT")

print("==========================================")


report = classification_report(

    actual,

    predicted,

    target_names=[
        "Normal",
        "Abnormal"
    ],

    zero_division=0

)


print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    actual,

    predicted

)


print("\n==========================================")

print("CONFUSION MATRIX")

print("==========================================")


print(
    cm
)


# ============================================================
# CREATE CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(
    figsize=(7, 6)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=[
        "Normal",
        "Abnormal"
    ],

    yticklabels=[
        "Normal",
        "Abnormal"
    ],

    cbar=False

)


plt.title(
    "Confusion Matrix - Chassis Anomaly Detection"
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "Actual Label"
)


plt.tight_layout()


plt.savefig(

    f"{OUTPUT_FOLDER}/confusion_matrix.png",

    dpi=150,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# CONDITION-LEVEL ANALYSIS
# ============================================================

print("\n==========================================")

print("CONDITION-LEVEL DETECTION")

print("==========================================")


condition_results = []


conditions = df[
    "Condition"
].unique()


for condition in conditions:

    condition_data = df[
        df["Condition"]
        == condition
    ]


    total = len(
        condition_data
    )


    anomalies = len(

        condition_data[
            condition_data[
                "ML_Status"
            ] == "Anomaly"
        ]

    )


    anomaly_rate = (

        anomalies
        / total
        * 100

    )


    condition_results.append({

        "Condition":
            condition,

        "Total_Windows":
            total,

        "Anomaly_Windows":
            anomalies,

        "Anomaly_Rate_Percent":
            anomaly_rate

    })


condition_df = pd.DataFrame(
    condition_results
)


print(

    condition_df.to_string(
        index=False
    )

)


# ============================================================
# SAVE CONDITION RESULTS
# ============================================================

condition_df.to_csv(

    f"{OUTPUT_FOLDER}/condition_detection.csv",

    index=False

)


# ============================================================
# SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1_Score"

    ],

    "Value": [

        accuracy,

        precision,

        recall,

        f1

    ]

})


metrics_df.to_csv(

    RESULT_FILE,

    index=False

)


# ============================================================
# CREATE METRIC GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.bar(

    metrics_df["Metric"],

    metrics_df["Value"]

)


plt.ylim(
    0,
    1.05
)


plt.title(
    "Machine Learning Model Performance"
)


plt.ylabel(
    "Score"
)


plt.xlabel(
    "Evaluation Metric"
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    f"{OUTPUT_FOLDER}/model_performance.png",

    dpi=150,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==========================================")

print(
    "PHASE 9 COMPLETED SUCCESSFULLY!"
)

print("==========================================")


print("\nGenerated files:")

print(
    "data/evaluation/model_metrics.csv"
)

print(
    "data/evaluation/condition_detection.csv"
)

print(
    "data/evaluation/confusion_matrix.png"
)

print(
    "data/evaluation/model_performance.png"
)