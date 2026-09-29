import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import joblib
import os


# ==========================================
# Configuration
# ==========================================

DATA_FILE = "data/vibration_dataset.csv"

MODEL_FOLDER = "models"

MODEL_FILE = "models/chassis_anomaly_model.pkl"

SCALER_FILE = "models/feature_scaler.pkl"

FEATURE_FILE = "data/ml_features.csv"


# ==========================================
# Create model folder
# ==========================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ==========================================
# Load dataset
# ==========================================

print("Loading vibration dataset...")

df = pd.read_csv(
    DATA_FILE
)

print("\nDataset loaded successfully!")

print(
    f"Total samples: {len(df)}"
)


# ==========================================
# Feature extraction function
# ==========================================

def extract_features(signal):

    # Mean
    mean_value = np.mean(signal)

    # Standard deviation
    std_value = np.std(signal)

    # RMS
    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    # Peak
    peak_value = np.max(
        np.abs(signal)
    )

    # --------------------------------------
    # Frequency analysis
    # --------------------------------------

    sample_rate = 100

    n = len(signal)

    # Remove average/DC component
    signal_centered = (
        signal - np.mean(signal)
    )

    # FFT
    fft_values = np.fft.fft(
        signal_centered
    )

    frequencies = np.fft.fftfreq(
        n,
        1 / sample_rate
    )

    magnitude = (
        np.abs(fft_values) / n
    )

    # Only positive frequencies
    positive = frequencies > 0

    positive_frequencies = (
        frequencies[positive]
    )

    positive_magnitude = (
        magnitude[positive]
    )

    # Dominant frequency
    dominant_index = np.argmax(
        positive_magnitude
    )

    dominant_frequency = (
        positive_frequencies[
            dominant_index
        ]
    )

    dominant_magnitude = (
        positive_magnitude[
            dominant_index
        ]
    )

    return [

        mean_value,

        std_value,

        rms_value,

        peak_value,

        dominant_frequency,

        dominant_magnitude
    ]


# ==========================================
# Create feature dataset
# ==========================================

print("\nExtracting features...")


conditions = df[
    "condition"
].unique()


features = []


for condition in conditions:

    condition_data = df[
        df["condition"] == condition
    ]

    signal = condition_data[
        "vibration"
    ].values

    feature_values = extract_features(
        signal
    )

    features.append({

        "Condition": condition,

        "Mean": feature_values[0],

        "Standard_Deviation":
            feature_values[1],

        "RMS":
            feature_values[2],

        "Peak":
            feature_values[3],

        "Dominant_Frequency":
            feature_values[4],

        "Dominant_Magnitude":
            feature_values[5]
    })


# ==========================================
# Create feature dataframe
# ==========================================

feature_df = pd.DataFrame(
    features
)


print("\nFeature dataset:")

print(
    feature_df.to_string(
        index=False
    )
)


# ==========================================
# Save feature dataset
# ==========================================

feature_df.to_csv(
    FEATURE_FILE,
    index=False
)

print(
    f"\nFeatures saved to: {FEATURE_FILE}"
)


# ==========================================
# Prepare ML data
# ==========================================

ml_features = [

    "Mean",

    "Standard_Deviation",

    "RMS",

    "Peak",

    "Dominant_Frequency",

    "Dominant_Magnitude"
]


X = feature_df[
    ml_features
]


# ==========================================
# Scale features
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ==========================================
# Create Isolation Forest model
# ==========================================

print(
    "\nTraining Isolation Forest model..."
)


model = IsolationForest(

    n_estimators=200,

    contamination=0.2,

    random_state=42
)


# ==========================================
# Train model
# ==========================================

model.fit(
    X_scaled
)


# ==========================================
# Predict anomalies
# ==========================================

predictions = model.predict(
    X_scaled
)


# Isolation Forest:
#
#  1  = Normal
# -1  = Anomaly


feature_df[
    "Prediction"
] = predictions


# ==========================================
# Convert prediction to readable status
# ==========================================

feature_df[
    "Status"
] = feature_df[
    "Prediction"
].map({

    1: "Normal",

    -1: "Anomaly"
})


# ==========================================
# Display results
# ==========================================

print(
    "\n=========================================="
)

print(
    "ANOMALY DETECTION RESULTS"
)

print(
    "=========================================="
)


print(
    feature_df[
        [
            "Condition",
            "RMS",
            "Peak",
            "Dominant_Frequency",
            "Status"
        ]
    ].to_string(
        index=False
    )
)


# ==========================================
# Save final results
# ==========================================

feature_df.to_csv(
    FEATURE_FILE,
    index=False
)


# ==========================================
# Save ML model
# ==========================================

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"\nModel saved to: {MODEL_FILE}"
)


# ==========================================
# Save scaler
# ==========================================

joblib.dump(
    scaler,
    SCALER_FILE
)

print(
    f"Scaler saved to: {SCALER_FILE}"
)


# ==========================================
# Final output
# ==========================================

print(
    "\n=========================================="
)

print(
    "PHASE 4 COMPLETED SUCCESSFULLY!"
)

print(
    "=========================================="
)

print(
    "\nMachine learning model trained successfully."
)

print(
    "The model can now be used for anomaly detection."
)