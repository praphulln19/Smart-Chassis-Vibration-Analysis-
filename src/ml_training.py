import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/vibration_dataset.csv"

WINDOW_SIZE = 200

SAMPLE_RATE = 100

FEATURE_FILE = "data/ml_window_features.csv"

SUMMARY_FILE = "data/ml_condition_summary.csv"

MODEL_FOLDER = "models"

MODEL_FILE = "models/window_anomaly_model.pkl"

SCALER_FILE = "models/window_feature_scaler.pkl"


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("==========================================")
print("WINDOW-BASED ML TRAINING")
print("==========================================")

print("\nLoading vibration dataset...")

df = pd.read_csv(
    DATA_FILE
)

print("\nDataset loaded successfully.")

print(
    f"Total samples: {len(df)}"
)

print(
    f"Window size: {WINDOW_SIZE} samples"
)

print(
    f"Sampling rate: {SAMPLE_RATE} Hz"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal):

    # --------------------------------------------------------
    # Time-domain features
    # --------------------------------------------------------

    mean_value = np.mean(signal)

    std_value = np.std(signal)

    rms_value = np.sqrt(
        np.mean(signal ** 2)
    )

    peak_value = np.max(
        np.abs(signal)
    )


    # --------------------------------------------------------
    # Frequency-domain features
    # --------------------------------------------------------

    signal_centered = (
        signal - mean_value
    )

    n = len(signal_centered)

    fft_values = np.fft.rfft(
        signal_centered
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / SAMPLE_RATE
    )

    magnitude = (
        np.abs(fft_values) / n
    )


    # Remove 0 Hz component
    if len(magnitude) > 1:

        frequencies = frequencies[1:]

        magnitude = magnitude[1:]


    # Dominant frequency
    dominant_index = np.argmax(
        magnitude
    )

    dominant_frequency = (
        frequencies[dominant_index]
    )

    dominant_magnitude = (
        magnitude[dominant_index]
    )


    return [

        mean_value,

        std_value,

        rms_value,

        peak_value,

        dominant_frequency,

        dominant_magnitude

    ]


# ============================================================
# CREATE WINDOWS
# ============================================================

print("\n==========================================")
print("CREATING VIBRATION WINDOWS")
print("==========================================")


conditions = df[
    "condition"
].unique()


window_records = []


for condition in conditions:

    print(
        f"\nProcessing: {condition}"
    )

    condition_data = df[
        df["condition"] == condition
    ]

    signal = condition_data[
        "vibration"
    ].values


    # Number of complete windows
    number_of_windows = (
        len(signal) // WINDOW_SIZE
    )


    print(
        f"Windows created: {number_of_windows}"
    )


    for window_number in range(
        number_of_windows
    ):

        start = (
            window_number
            * WINDOW_SIZE
        )

        end = (
            start
            + WINDOW_SIZE
        )


        # Extract window
        window = signal[
            start:end
        ]


        # Extract features
        features = extract_features(
            window
        )


        # Store window information
        window_records.append({

            "Condition":
                condition,

            "Window":
                window_number,

            "Mean":
                features[0],

            "Standard_Deviation":
                features[1],

            "RMS":
                features[2],

            "Peak":
                features[3],

            "Dominant_Frequency":
                features[4],

            "Dominant_Magnitude":
                features[5]
        })


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

feature_df = pd.DataFrame(
    window_records
)


print("\n==========================================")
print("WINDOW FEATURE DATASET")
print("==========================================")

print(
    f"Total windows: {len(feature_df)}"
)

print(
    "\nWindows per condition:"
)

print(
    feature_df[
        "Condition"
    ].value_counts()
)


# ============================================================
# SAVE RAW WINDOW FEATURES
# ============================================================

feature_df.to_csv(
    FEATURE_FILE,
    index=False
)

print(
    f"\nWindow features saved to:"
)

print(
    FEATURE_FILE
)


# ============================================================
# SELECT ML FEATURES
# ============================================================

ML_FEATURES = [

    "Mean",

    "Standard_Deviation",

    "RMS",

    "Peak",

    "Dominant_Frequency",

    "Dominant_Magnitude"

]


X = feature_df[
    ML_FEATURES
]


# ============================================================
# TRAIN ONLY ON NORMAL DATA
# ============================================================

print("\n==========================================")
print("TRAINING ANOMALY DETECTOR")
print("==========================================")


normal_data = feature_df[
    feature_df["Condition"]
    == "Normal"
]


X_normal = normal_data[
    ML_FEATURES
]


print(
    f"Normal training windows: {len(X_normal)}"
)


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_normal_scaled = scaler.fit_transform(
    X_normal
)


# ============================================================
# CREATE ISOLATION FOREST
# ============================================================

model = IsolationForest(

    n_estimators=200,

    contamination=0.10,

    random_state=42

)


# ============================================================
# TRAIN MODEL
# ============================================================

print(
    "\nTraining Isolation Forest..."
)

model.fit(
    X_normal_scaled
)


print(
    "Model training completed."
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)


print(
    f"\nModel saved to:"
)

print(
    MODEL_FILE
)

print(
    f"\nScaler saved to:"
)

print(
    SCALER_FILE
)


# ============================================================
# PREDICT ALL WINDOWS
# ============================================================

print("\n==========================================")
print("TESTING ALL VIBRATION WINDOWS")
print("==========================================")


X_all = feature_df[
    ML_FEATURES
]


X_all_scaled = scaler.transform(
    X_all
)


predictions = model.predict(
    X_all_scaled
)


# Isolation Forest:
#
# 1  = Normal
# -1 = Anomaly


feature_df[
    "Prediction"
] = predictions


feature_df[
    "ML_Status"
] = feature_df[
    "Prediction"
].map({

    1: "Normal",

    -1: "Anomaly"

})


# ============================================================
# ANOMALY SCORE
# ============================================================

decision_scores = (
    model.decision_function(
        X_all_scaled
    )
)


feature_df[
    "Anomaly_Score"
] = decision_scores


# ============================================================
# DISPLAY WINDOW RESULTS
# ============================================================

print(
    "\nFirst 20 window predictions:"
)

print(

    feature_df[
        [
            "Condition",
            "Window",
            "RMS",
            "Peak",
            "Dominant_Frequency",
            "ML_Status"
        ]
    ].head(20).to_string(
        index=False
    )

)


# ============================================================
# SAVE WINDOW RESULTS
# ============================================================

feature_df.to_csv(
    FEATURE_FILE,
    index=False
)


# ============================================================
# CONDITION-LEVEL SUMMARY
# ============================================================

print("\n==========================================")
print("CONDITION-LEVEL ML SUMMARY")
print("==========================================")


summary_records = []


for condition in conditions:

    condition_windows = feature_df[
        feature_df["Condition"]
        == condition
    ]


    total_windows = len(
        condition_windows
    )


    anomaly_windows = len(
        condition_windows[
            condition_windows[
                "ML_Status"
            ] == "Anomaly"
        ]
    )


    anomaly_percentage = (

        anomaly_windows
        / total_windows
        * 100

    )


    mean_rms = (
        condition_windows[
            "RMS"
        ].mean()
    )


    mean_peak = (
        condition_windows[
            "Peak"
        ].mean()
    )


    mean_frequency = (
        condition_windows[
            "Dominant_Frequency"
        ].mean()
    )


    summary_records.append({

        "Condition":
            condition,

        "Total_Windows":
            total_windows,

        "Anomaly_Windows":
            anomaly_windows,

        "Anomaly_Percentage":
            anomaly_percentage,

        "Mean_RMS":
            mean_rms,

        "Mean_Peak":
            mean_peak,

        "Mean_Dominant_Frequency":
            mean_frequency

    })


summary_df = pd.DataFrame(
    summary_records
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(

    summary_df.to_string(
        index=False
    )

)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    f"\nCondition summary saved to:"
)

print(
    SUMMARY_FILE
)


# ============================================================
# EVALUATION
# ============================================================

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")


# Actual label:
#
# Normal = 0
# Everything else = 1


actual_labels = feature_df[
    "Condition"
].apply(

    lambda x:
        0 if x == "Normal"
        else 1

).values


# Predicted label:
#
# Normal = 0
# Anomaly = 1


predicted_labels = feature_df[
    "ML_Status"
].apply(

    lambda x:
        0 if x == "Normal"
        else 1

).values


print(
    "\nClassification Report:"
)

print(

    classification_report(

        actual_labels,

        predicted_labels,

        target_names=[
            "Normal",
            "Abnormal"
        ],

        zero_division=0

    )

)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        actual_labels,
        predicted_labels
    )
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("PHASE 7 COMPLETED SUCCESSFULLY!")
print("==========================================")


print(
    "\nGenerated files:"
)

print(
    "1. data/ml_window_features.csv"
)

print(
    "2. data/ml_condition_summary.csv"
)

print(
    "3. models/window_anomaly_model.pkl"
)

print(
    "4. models/window_feature_scaler.pkl"
)

print(
    "\nWindow-based anomaly detection is ready."
)