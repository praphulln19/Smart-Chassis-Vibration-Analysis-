import numpy as np
import pandas as pd
import os


# -----------------------------------
# Configuration
# -----------------------------------

SAMPLE_RATE = 100
DURATION = 60

np.random.seed(42)

time = np.arange(
    0,
    DURATION,
    1 / SAMPLE_RATE
)


# -----------------------------------
# Normal vibration
# -----------------------------------

def generate_normal():

    signal = (
        0.15 * np.sin(2 * np.pi * 5 * time)
        + 0.05 * np.random.randn(len(time))
    )

    return signal


# -----------------------------------
# Rough road vibration
# -----------------------------------

def generate_rough_road():

    signal = (
        0.35 * np.sin(2 * np.pi * 8 * time)
        + 0.15 * np.random.randn(len(time))
    )

    return signal


# -----------------------------------
# High vibration
# -----------------------------------

def generate_high_vibration():

    signal = (
        0.8 * np.sin(2 * np.pi * 15 * time)
        + 0.25 * np.random.randn(len(time))
    )

    return signal


# -----------------------------------
# Loose joint
# -----------------------------------

def generate_loose_joint():

    signal = (
        0.25 * np.sin(2 * np.pi * 6 * time)
        + 0.65 * np.sin(2 * np.pi * 22 * time)
        + 0.12 * np.random.randn(len(time))
    )

    return signal


# -----------------------------------
# Structural anomaly
# -----------------------------------

def generate_structural_anomaly():

    signal = (
        0.2 * np.sin(2 * np.pi * 5 * time)
        + 1.2 * np.sin(2 * np.pi * 25 * time)
        + 0.35 * np.random.randn(len(time))
    )

    return signal


# -----------------------------------
# Generate all conditions
# -----------------------------------

conditions = {
    "Normal": generate_normal(),
    "Rough_Road": generate_rough_road(),
    "High_Vibration": generate_high_vibration(),
    "Loose_Joint": generate_loose_joint(),
    "Structural_Anomaly": generate_structural_anomaly()
}


# -----------------------------------
# Create data folder
# -----------------------------------

os.makedirs("data", exist_ok=True)


# -----------------------------------
# Save individual datasets
# -----------------------------------

for condition, vibration in conditions.items():

    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "condition": condition
    })

    file_path = f"data/{condition.lower()}.csv"

    df.to_csv(
        file_path,
        index=False
    )

    print(f"Saved: {file_path}")


# -----------------------------------
# Create combined dataset
# -----------------------------------

combined = pd.concat(
    [
        pd.DataFrame({
            "time": time,
            "vibration": vibration,
            "condition": condition
        })
        for condition, vibration in conditions.items()
    ],
    ignore_index=True
)


# -----------------------------------
# Save combined dataset
# -----------------------------------

combined.to_csv(
    "data/vibration_dataset.csv",
    index=False
)


# -----------------------------------
# Final output
# -----------------------------------

print("\nCombined dataset created successfully!")

print("\nFirst 5 rows:")
print(combined.head())

print("\nDataset shape:")
print(combined.shape)

print("\nConditions:")
print(combined["condition"].unique())