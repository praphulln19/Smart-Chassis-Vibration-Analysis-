import pandas as pd
import numpy as np


# ==========================================
# Configuration
# ==========================================

FEATURE_FILE = "data/ml_features.csv"

OUTPUT_FILE = "data/risk_assessment.csv"


# ==========================================
# Load ML feature data
# ==========================================

print("Loading ML feature data...")

df = pd.read_csv(FEATURE_FILE)

print("\nData loaded successfully!")

print("\nAvailable conditions:")

print(
    df["Condition"].tolist()
)


# ==========================================
# Calculate normalized scores
# ==========================================

# We use the maximum values in our dataset
# to normalize the measurements.

max_rms = df["RMS"].max()

max_peak = df["Peak"].max()

max_frequency = df[
    "Dominant_Frequency"
].max()


# Avoid division by zero
if max_rms == 0:
    max_rms = 1

if max_peak == 0:
    max_peak = 1

if max_frequency == 0:
    max_frequency = 1


# ==========================================
# Risk calculation function
# ==========================================

def calculate_risk(row):

    # --------------------------------------
    # Normalize individual measurements
    # --------------------------------------

    rms_score = (
        row["RMS"] / max_rms
    )

    peak_score = (
        row["Peak"] / max_peak
    )

    frequency_score = (
        row["Dominant_Frequency"]
        / max_frequency
    )


    # --------------------------------------
    # Calculate combined risk score
    # --------------------------------------

    risk_score = (

        0.40 * rms_score

        + 0.35 * peak_score

        + 0.25 * frequency_score

    )


    # --------------------------------------
    # Consider ML anomaly result
    # --------------------------------------

    if row["Status"] == "Anomaly":

        risk_score += 0.20


    # Limit score to 1
    risk_score = min(
        risk_score,
        1.0
    )


    # --------------------------------------
    # Classify risk
    # --------------------------------------

    if risk_score < 0.40:

        risk_level = "NORMAL"

        telltale = "SAFE"

        recommendation = (
            "Vibration pattern is within "
            "the simulated normal range."
        )


    elif risk_score < 0.70:

        risk_level = "WARNING"

        telltale = "CHECK CHASSIS"

        recommendation = (
            "Abnormal vibration detected. "
            "Inspect chassis and joints."
        )


    else:

        risk_level = "CRITICAL"

        telltale = "STOP / INSPECT"

        recommendation = (
            "High abnormal vibration pattern. "
            "Detailed chassis inspection recommended."
        )


    return (

        risk_score,

        risk_level,

        telltale,

        recommendation

    )


# ==========================================
# Apply risk assessment
# ==========================================

results = []


for index, row in df.iterrows():

    (

        risk_score,

        risk_level,

        telltale,

        recommendation

    ) = calculate_risk(row)


    results.append({

        "Condition":
            row["Condition"],

        "RMS":
            row["RMS"],

        "Peak":
            row["Peak"],

        "Dominant_Frequency_Hz":
            row["Dominant_Frequency"],

        "ML_Status":
            row["Status"],

        "Risk_Score":
            risk_score,

        "Risk_Level":
            risk_level,

        "Safety_Telltale":
            telltale,

        "Recommendation":
            recommendation
    })


# ==========================================
# Create final dataframe
# ==========================================

risk_df = pd.DataFrame(
    results
)


# ==========================================
# Display results
# ==========================================

print("\n==========================================")

print("SMART CHASSIS RISK ASSESSMENT")

print("==========================================")


print(

    risk_df[
        [
            "Condition",
            "RMS",
            "Peak",
            "Dominant_Frequency_Hz",
            "ML_Status",
            "Risk_Score",
            "Risk_Level",
            "Safety_Telltale"
        ]
    ].to_string(
        index=False
    )

)


# ==========================================
# Save results
# ==========================================

risk_df.to_csv(

    OUTPUT_FILE,

    index=False

)


print("\nRisk assessment saved to:")

print(OUTPUT_FILE)


# ==========================================
# Final message
# ==========================================

print(
    "\n=========================================="
)

print(
    "PHASE 5 COMPLETED SUCCESSFULLY!"
)

print(
    "=========================================="
)

print(
    "\nSafety telltale system is ready."
)