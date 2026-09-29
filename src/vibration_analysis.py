import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ==========================================
# Configuration
# ==========================================

DATA_FILE = "data/vibration_dataset.csv"
OUTPUT_FILE = "data/vibration_features.csv"

# Number of samples to display in time-domain graphs
PLOT_SAMPLES = 2000


# ==========================================
# Load Dataset
# ==========================================

print("Loading vibration dataset...")

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nConditions:")
print(df["condition"].unique())


# ==========================================
# Function to calculate vibration features
# ==========================================

def calculate_features(signal):

    # Mean
    mean_value = np.mean(signal)

    # Standard deviation
    std_value = np.std(signal)

    # RMS
    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    # Maximum absolute vibration
    peak_value = np.max(
        np.abs(signal)
    )

    # Minimum value
    minimum_value = np.min(signal)

    # Maximum value
    maximum_value = np.max(signal)

    return {
        "Mean": mean_value,
        "Standard_Deviation": std_value,
        "RMS": rms_value,
        "Peak": peak_value,
        "Minimum": minimum_value,
        "Maximum": maximum_value
    }


# ==========================================
# Analyze each chassis condition
# ==========================================

print("\n==========================================")
print("VIBRATION ANALYSIS")
print("==========================================")

conditions = df["condition"].unique()

results = []


for condition in conditions:

    print(f"\nAnalyzing: {condition}")

    # Select data for current condition
    condition_data = df[
        df["condition"] == condition
    ]

    # Get vibration signal
    signal = condition_data[
        "vibration"
    ].values

    # Calculate features
    features = calculate_features(signal)

    # Add condition name
    features["Condition"] = condition

    # Store result
    results.append(features)


# ==========================================
# Create feature dataframe
# ==========================================

feature_df = pd.DataFrame(results)


# Reorder columns
feature_df = feature_df[
    [
        "Condition",
        "Mean",
        "Standard_Deviation",
        "RMS",
        "Peak",
        "Minimum",
        "Maximum"
    ]
]


# ==========================================
# Display results
# ==========================================

print("\n==========================================")
print("VIBRATION FEATURE RESULTS")
print("==========================================")

print(
    feature_df.to_string(
        index=False
    )
)


# ==========================================
# Save results
# ==========================================

feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFeature data saved to:")
print(OUTPUT_FILE)


# ==========================================
# Create output folder for graphs
# ==========================================

os.makedirs(
    "data/graphs",
    exist_ok=True
)


# ==========================================
# Generate vibration graphs
# ==========================================

print("\n==========================================")
print("GENERATING VIBRATION GRAPHS")
print("==========================================")


for condition in conditions:

    condition_data = df[
        df["condition"] == condition
    ]

    # Select first PLOT_SAMPLES
    plot_data = condition_data.head(
        PLOT_SAMPLES
    )

    time = plot_data["time"].values

    vibration = plot_data[
        "vibration"
    ].values

    # Create figure
    plt.figure(
        figsize=(12, 5)
    )

    # Plot vibration
    plt.plot(
        time,
        vibration
    )

    # Title
    plt.title(
        f"Chassis Vibration - {condition}"
    )

    # Axis labels
    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Vibration"
    )

    # Grid
    plt.grid(
        True,
        alpha=0.3
    )

    # Save graph
    filename = (
        condition.lower()
        + "_vibration.png"
    )

    filepath = os.path.join(
        "data/graphs",
        filename
    )

    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Saved graph: {filepath}"
    )

    # Display graph
    plt.show()

    # Close figure
    plt.close()


# ==========================================
# RMS Comparison Graph
# ==========================================

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    feature_df["Condition"],
    feature_df["RMS"]
)

plt.title(
    "RMS Vibration Comparison"
)

plt.xlabel(
    "Chassis Condition"
)

plt.ylabel(
    "RMS Vibration"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "data/graphs/rms_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ==========================================
# Peak Vibration Comparison
# ==========================================

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    feature_df["Condition"],
    feature_df["Peak"]
)

plt.title(
    "Peak Vibration Comparison"
)

plt.xlabel(
    "Chassis Condition"
)

plt.ylabel(
    "Peak Vibration"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "data/graphs/peak_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ==========================================
# Final message
# ==========================================

print("\n==========================================")
print("PHASE 2 COMPLETED SUCCESSFULLY!")
print("==========================================")

print("\nGenerated files:")

print("1. data/vibration_features.csv")

print("2. data/graphs/")

print("\nVibration analysis is complete.")