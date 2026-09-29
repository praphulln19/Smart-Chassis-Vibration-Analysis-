import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.fft import fft, fftfreq


# ==========================================
# Configuration
# ==========================================

DATA_FILE = "data/vibration_dataset.csv"

OUTPUT_FILE = "data/frequency_features.csv"

GRAPH_FOLDER = "data/graphs"

SAMPLE_RATE = 100


# ==========================================
# Create graph folder
# ==========================================

os.makedirs(
    GRAPH_FOLDER,
    exist_ok=True
)


# ==========================================
# Load Dataset
# ==========================================

print("Loading vibration dataset...")

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nConditions:")
print(df["condition"].unique())


# ==========================================
# Function: FFT Analysis
# ==========================================

def analyze_frequency(signal):

    # Number of samples
    n = len(signal)

    # Remove DC/average component
    signal = signal - np.mean(signal)

    # Perform FFT
    fft_values = fft(signal)

    # Calculate frequency values
    frequencies = fftfreq(
        n,
        1 / SAMPLE_RATE
    )

    # Calculate magnitude
    magnitude = np.abs(fft_values) / n

    # Only use positive frequencies
    positive = frequencies >= 0

    positive_frequencies = frequencies[positive]

    positive_magnitude = magnitude[positive]

    # Ignore frequency 0 Hz
    if len(positive_frequencies) > 1:

        positive_frequencies = positive_frequencies[1:]

        positive_magnitude = positive_magnitude[1:]

    # Find dominant frequency
    dominant_index = np.argmax(
        positive_magnitude
    )

    dominant_frequency = (
        positive_frequencies[dominant_index]
    )

    dominant_magnitude = (
        positive_magnitude[dominant_index]
    )

    return (
        positive_frequencies,
        positive_magnitude,
        dominant_frequency,
        dominant_magnitude
    )


# ==========================================
# Analyze every condition
# ==========================================

conditions = df["condition"].unique()

frequency_results = []


print("\n==========================================")
print("FREQUENCY ANALYSIS")
print("==========================================")


for condition in conditions:

    print(
        f"\nAnalyzing frequency: {condition}"
    )

    # Select condition data
    condition_data = df[
        df["condition"] == condition
    ]

    # Get vibration signal
    signal = condition_data[
        "vibration"
    ].values

    # Perform FFT
    (
        frequencies,
        magnitude,
        dominant_frequency,
        dominant_magnitude
    ) = analyze_frequency(signal)

    # Store result
    frequency_results.append({

        "Condition": condition,

        "Dominant_Frequency_Hz":
            dominant_frequency,

        "Dominant_Magnitude":
            dominant_magnitude
    })


    # ======================================
    # Plot Frequency Spectrum
    # ======================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        frequencies,
        magnitude
    )

    plt.title(
        f"Frequency Spectrum - {condition}"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude"
    )

    # Show useful frequency range
    plt.xlim(
        0,
        50
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    # Create filename
    filename = (
        condition.lower()
        + "_frequency.png"
    )

    filepath = os.path.join(
        GRAPH_FOLDER,
        filename
    )

    # Save graph
    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Saved graph: {filepath}"
    )

    plt.show()

    plt.close()


# ==========================================
# Create Frequency Feature DataFrame
# ==========================================

frequency_df = pd.DataFrame(
    frequency_results
)


# ==========================================
# Display results
# ==========================================

print("\n==========================================")
print("DOMINANT FREQUENCY RESULTS")
print("==========================================")

print(
    frequency_df.to_string(
        index=False
    )
)


# ==========================================
# Save frequency results
# ==========================================

frequency_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nFrequency features saved to:"
)

print(
    OUTPUT_FILE
)


# ==========================================
# Dominant Frequency Comparison
# ==========================================

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    frequency_df["Condition"],
    frequency_df[
        "Dominant_Frequency_Hz"
    ]
)

plt.title(
    "Dominant Frequency Comparison"
)

plt.xlabel(
    "Chassis Condition"
)

plt.ylabel(
    "Dominant Frequency (Hz)"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# Save comparison graph

plt.savefig(
    "data/graphs/dominant_frequency_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ==========================================
# Final Output
# ==========================================

print("\n==========================================")
print("PHASE 3 COMPLETED SUCCESSFULLY!")
print("==========================================")

print("\nGenerated file:")

print(
    "data/frequency_features.csv"
)

print("\nGenerated graphs:")

print(
    "data/graphs/"
)

print(
    "\nFrequency analysis is complete!"
)