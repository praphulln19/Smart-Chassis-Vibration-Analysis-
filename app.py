import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Chassis Analyzer",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/vibration_dataset.csv"

WINDOW_FEATURE_FILE = "data/ml_window_features.csv"

SUMMARY_FILE = "data/ml_condition_summary.csv"

METRICS_FILE = "data/evaluation/model_metrics.csv"

CONDITION_EVALUATION_FILE = (
    "data/evaluation/condition_detection.csv"
)

CONFUSION_MATRIX_FILE = (
    "data/evaluation/confusion_matrix.png"
)

MODEL_PERFORMANCE_FILE = (
    "data/evaluation/model_performance.png"
)

SAMPLE_RATE = 100

DISPLAY_SAMPLES = 2000


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_project_data():

    vibration_df = pd.read_csv(
        DATA_FILE
    )

    window_df = pd.read_csv(
        WINDOW_FEATURE_FILE
    )

    summary_df = pd.read_csv(
        SUMMARY_FILE
    )

    metrics_df = pd.read_csv(
        METRICS_FILE
    )

    condition_evaluation_df = pd.read_csv(
        CONDITION_EVALUATION_FILE
    )

    return (
        vibration_df,
        window_df,
        summary_df,
        metrics_df,
        condition_evaluation_df
    )


# ============================================================
# LOAD DATA WITH ERROR HANDLING
# ============================================================

try:

    (
        vibration_df,
        window_df,
        summary_df,
        metrics_df,
        condition_evaluation_df
    ) = load_project_data()

except Exception as error:

    st.error(
        "Project data could not be loaded."
    )

    st.warning(
        """
        Make sure you have already run:

        python src/ml_training.py

        and:

        python src/model_evaluation.py
        """
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🚗 Smart Chassis"
)

st.sidebar.markdown(
    "### Navigation"
)


page = st.sidebar.radio(

    "Go to",

    [
        "🏠 Dashboard",
        "📊 Vibration Analysis",
        "🤖 ML Analysis",
        "📈 Model Performance",
        "📋 Dataset",
        "ℹ️ About Project"
    ]

)


st.sidebar.divider()


st.sidebar.info(
    """
    **Software Simulation Prototype**

    The current system uses simulated
    vibration data for demonstrating
    vibration-based anomaly detection.
    """
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚗 Smart Chassis Vibration Analysis System"
)

st.markdown(
    """
    **Intelligent Vibration Monitoring & Structural Risk Analysis**
    """
)

st.caption(
    """
    The system analyzes vibration signals, extracts
    time/frequency features, detects abnormal patterns,
    and presents a safety-oriented indication.
    """
)

st.divider()


# ============================================================
# CONDITION SELECTOR
# ============================================================

conditions = vibration_df[
    "condition"
].unique()


selected_condition = st.selectbox(

    "Select Chassis Condition",

    conditions

)


# ============================================================
# SELECT CONDITION DATA
# ============================================================

selected_signal_df = vibration_df[
    vibration_df["condition"]
    == selected_condition
].copy()


selected_windows = window_df[
    window_df["Condition"]
    == selected_condition
].copy()


# ============================================================
# CHECK SELECTED DATA
# ============================================================

if selected_windows.empty:

    st.error(
        "No ML window data found for the selected condition."
    )

    st.stop()


# ============================================================
# BASIC FEATURES
# ============================================================

rms_value = selected_windows[
    "RMS"
].mean()


peak_value = selected_windows[
    "Peak"
].mean()


dominant_frequency = selected_windows[
    "Dominant_Frequency"
].mean()


total_windows = len(
    selected_windows
)


anomaly_windows = len(

    selected_windows[
        selected_windows["ML_Status"]
        == "Anomaly"
    ]

)


if total_windows > 0:

    anomaly_percentage = (

        anomaly_windows
        / total_windows
        * 100

    )

else:

    anomaly_percentage = 0


# ============================================================
# RISK LEVEL
# ============================================================

if anomaly_percentage < 20:

    risk_level = "NORMAL"

    telltale = "SAFE"

    recommendation = (
        "Vibration pattern is close to "
        "the simulated normal baseline."
    )


elif anomaly_percentage < 50:

    risk_level = "WARNING"

    telltale = "CHECK CHASSIS"

    recommendation = (
        "Some vibration windows differ "
        "from the normal baseline. "
        "Further inspection is recommended."
    )


else:

    risk_level = "CRITICAL"

    telltale = "STOP / INSPECT"

    recommendation = (
        "A large proportion of vibration "
        "windows differs from the normal "
        "baseline. Detailed inspection "
        "is recommended."
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "🏠 Dashboard":

    st.subheader(
        "📊 Current Chassis Status"
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Average RMS",
            f"{rms_value:.3f}"
        )


    with col2:

        st.metric(
            "Average Peak",
            f"{peak_value:.3f}"
        )


    with col3:

        st.metric(
            "Dominant Frequency",
            f"{dominant_frequency:.2f} Hz"
        )


    with col4:

        st.metric(
            "Anomaly Windows",
            f"{anomaly_windows}/{total_windows}"
        )


    st.divider()


    # --------------------------------------------------------
    # SAFETY STATUS
    # --------------------------------------------------------

    st.subheader(
        "🚨 Safety Telltale"
    )


    if risk_level == "NORMAL":

        st.success(
            f"🟢 CHASSIS STATUS: {risk_level}"
        )


    elif risk_level == "WARNING":

        st.warning(
            f"🟡 CHASSIS STATUS: {risk_level}"
        )


    else:

        st.error(
            f"🔴 CHASSIS STATUS: {risk_level}"
        )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            f"**Safety Telltale:** {telltale}"
        )


    with col2:

        st.info(
            f"**Anomaly Percentage:** "
            f"{anomaly_percentage:.1f}%"
        )


    st.write(
        f"**Recommendation:** {recommendation}"
    )


    st.divider()


    # --------------------------------------------------------
    # VIBRATION SIGNAL
    # --------------------------------------------------------

    st.subheader(
        "📈 Vibration Signal"
    )


    signal = selected_signal_df[
        "vibration"
    ].values[:DISPLAY_SAMPLES]


    time = selected_signal_df[
        "time"
    ].values[:DISPLAY_SAMPLES]


    fig, ax = plt.subplots(
        figsize=(12, 4)
    )


    ax.plot(
        time,
        signal
    )


    ax.set_title(
        f"Vibration Signal - "
        f"{selected_condition}"
    )


    ax.set_xlabel(
        "Time (seconds)"
    )


    ax.set_ylabel(
        "Vibration"
    )


    ax.grid(
        True,
        alpha=0.3
    )


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


    # --------------------------------------------------------
    # FREQUENCY SPECTRUM
    # --------------------------------------------------------

    st.subheader(
        "🔊 Frequency Spectrum"
    )


    full_signal = selected_signal_df[
        "vibration"
    ].values


    centered_signal = (
        full_signal
        - np.mean(full_signal)
    )


    n = len(
        centered_signal
    )


    frequencies = rfftfreq(
        n,
        d=1 / SAMPLE_RATE
    )


    fft_values = np.abs(
        rfft(centered_signal)
    ) / n


    fig2, ax2 = plt.subplots(
        figsize=(12, 4)
    )


    ax2.plot(
        frequencies,
        fft_values
    )


    ax2.set_xlim(
        0,
        50
    )


    ax2.set_title(
        f"Frequency Spectrum - "
        f"{selected_condition}"
    )


    ax2.set_xlabel(
        "Frequency (Hz)"
    )


    ax2.set_ylabel(
        "Magnitude"
    )


    ax2.grid(
        True,
        alpha=0.3
    )


    st.pyplot(
        fig2,
        use_container_width=True
    )


    plt.close(fig2)


# ============================================================
# VIBRATION ANALYSIS PAGE
# ============================================================

elif page == "📊 Vibration Analysis":

    st.header(
        "📊 Vibration Analysis"
    )


    st.write(
        f"Selected condition: **{selected_condition}**"
    )


    st.divider()


    # --------------------------------------------------------
    # STATISTICAL FEATURES
    # --------------------------------------------------------

    st.subheader(
        "📊 Statistical Features"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "RMS",
            f"{rms_value:.4f}"
        )


    with col2:

        st.metric(
            "Peak",
            f"{peak_value:.4f}"
        )


    with col3:

        st.metric(
            "Dominant Frequency",
            f"{dominant_frequency:.2f} Hz"
        )


    st.divider()


    # --------------------------------------------------------
    # WINDOW FEATURES
    # --------------------------------------------------------

    st.subheader(
        "🪟 Window-Level Features"
    )


    display_columns = [

        "Window",

        "RMS",

        "Peak",

        "Dominant_Frequency",

        "Dominant_Magnitude",

        "ML_Status",

        "Anomaly_Score"

    ]


    st.dataframe(

        selected_windows[
            display_columns
        ].round(4),

        use_container_width=True

    )


# ============================================================
# ML ANALYSIS PAGE
# ============================================================

elif page == "🤖 ML Analysis":

    st.header(
        "🤖 Machine Learning Analysis"
    )


    st.markdown(
        """
        The anomaly detector is trained using vibration
        windows representing the simulated **Normal**
        condition.

        Each vibration window is represented using
        time-domain and frequency-domain features.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # ML PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Machine Learning Pipeline"
    )


    st.markdown(
        """
        **Vibration Signal**

        ↓

        **Window Segmentation**

        ↓

        **Feature Extraction**

        ↓

        **Standardization**

        ↓

        **Isolation Forest**

        ↓

        **Normal / Anomaly**
        """
    )


    st.divider()


    # --------------------------------------------------------
    # ML METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Windows",
            total_windows
        )


    with col2:

        st.metric(
            "Anomalous Windows",
            anomaly_windows
        )


    with col3:

        st.metric(
            "Anomaly Rate",
            f"{anomaly_percentage:.1f}%"
        )


    with col4:

        most_common_status = (
            selected_windows[
                "ML_Status"
            ]
            .value_counts()
            .idxmax()
        )


        st.metric(
            "ML Status",
            most_common_status
        )


    st.divider()


    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📊 ML Prediction Distribution"
    )


    prediction_counts = (
        selected_windows[
            "ML_Status"
        ]
        .value_counts()
    )


    st.bar_chart(
        prediction_counts
    )


    st.divider()


    # --------------------------------------------------------
    # WINDOW PREDICTIONS
    # --------------------------------------------------------

    st.subheader(
        "🔍 Window-Level ML Predictions"
    )


    st.dataframe(

        selected_windows[
            [
                "Condition",
                "Window",
                "RMS",
                "Peak",
                "Dominant_Frequency",
                "ML_Status",
                "Anomaly_Score"
            ]
        ].round(4),

        use_container_width=True

    )


    st.divider()


    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "💡 How the ML Result Is Produced"
    )


    st.write(
        """
        The Isolation Forest model learns the characteristics
        of the simulated Normal vibration windows.

        During analysis, each vibration window is compared
        against this learned normal pattern.

        Windows that differ significantly from the normal
        pattern are classified as anomalies.
        """
    )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📈 Model Performance":

    st.header(
        "📈 Machine Learning Model Performance"
    )


    st.markdown(
        """
        This section presents the evaluation results of the
        window-based Isolation Forest anomaly detection model
        using the simulated vibration dataset.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # GET METRICS
    # --------------------------------------------------------

    accuracy = metrics_df.loc[
        metrics_df["Metric"] == "Accuracy",
        "Value"
    ].iloc[0]


    precision = metrics_df.loc[
        metrics_df["Metric"] == "Precision",
        "Value"
    ].iloc[0]


    recall = metrics_df.loc[
        metrics_df["Metric"] == "Recall",
        "Value"
    ].iloc[0]


    f1_score = metrics_df.loc[
        metrics_df["Metric"] == "F1_Score",
        "Value"
    ].iloc[0]


    # --------------------------------------------------------
    # PERFORMANCE CARDS
    # --------------------------------------------------------

    st.subheader(
        "📊 Evaluation Metrics"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{f1_score * 100:.2f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "🔎 Performance Interpretation"
    )


    st.write(
        f"""
        On the simulated evaluation dataset, the model achieved:

        - **Accuracy:** {accuracy * 100:.2f}%
        - **Precision:** {precision * 100:.2f}%
        - **Recall:** {recall * 100:.2f}%
        - **F1 Score:** {f1_score * 100:.2f}%

        The model detected all abnormal windows in the current
        simulated evaluation dataset. Some Normal windows were
        also flagged as anomalous, resulting in a 10% false-positive
        rate for the Normal condition.
        """
    )


    st.info(
        """
        These performance values are specific to the current
        simulated dataset. They should not be interpreted as
        certified real-world vehicle diagnostic performance.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # MODEL PERFORMANCE CHART
    # --------------------------------------------------------

    st.subheader(
        "📊 Performance Metrics Chart"
    )


    performance_data = metrics_df.copy()


    performance_data[
        "Percentage"
    ] = (
        performance_data[
            "Value"
        ] * 100
    )


    st.bar_chart(

        performance_data.set_index(
            "Metric"
        )[
            "Percentage"
        ]

    )


    st.divider()


    # --------------------------------------------------------
    # CONDITION-LEVEL RESULTS
    # --------------------------------------------------------

    st.subheader(
        "🔍 Condition-Level Detection"
    )


    condition_display = (
        condition_evaluation_df.copy()
    )


    condition_display[
        "Anomaly_Rate_Percent"
    ] = condition_display[
        "Anomaly_Rate_Percent"
    ].round(2)


    condition_display = (
        condition_display.rename(

            columns={

                "Total_Windows":
                    "Total Windows",

                "Anomaly_Windows":
                    "Anomalous Windows",

                "Anomaly_Rate_Percent":
                    "Anomaly Rate (%)"

            }

        )
    )


    st.dataframe(

        condition_display,

        use_container_width=True

    )


    st.divider()


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "🧩 Confusion Matrix"
    )


    try:

        st.image(

            CONFUSION_MATRIX_FILE,

            caption=(
                "Confusion Matrix - "
                "Chassis Anomaly Detection"
            ),

            use_container_width=True

        )

    except Exception:

        st.warning(
            "Confusion matrix image could not be displayed."
        )


    st.divider()


    # --------------------------------------------------------
    # SAVED PERFORMANCE GRAPH
    # --------------------------------------------------------

    st.subheader(
        "📈 Saved Model Performance Graph"
    )


    try:

        st.image(

            MODEL_PERFORMANCE_FILE,

            caption=(
                "Machine Learning Model Performance"
            ),

            use_container_width=True

        )

    except Exception:

        st.warning(
            "Model performance graph could not be displayed."
        )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📋 Dataset":

    st.header(
        "📋 Vibration Dataset"
    )


    # --------------------------------------------------------
    # DATASET SIZE
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Vibration Samples",
            f"{len(vibration_df):,}"
        )


    with col2:

        st.metric(
            "ML Windows",
            len(window_df)
        )


    with col3:

        st.metric(
            "Conditions",
            len(conditions)
        )


    st.divider()


    # --------------------------------------------------------
    # CONDITION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📊 Dataset Distribution"
    )


    condition_counts = (
        vibration_df[
            "condition"
        ]
        .value_counts()
    )


    st.bar_chart(
        condition_counts
    )


    st.divider()


    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    st.subheader(
        "📋 Raw Sensor Data"
    )


    st.dataframe(

        vibration_df.head(500),

        use_container_width=True

    )


    st.divider()


    # --------------------------------------------------------
    # ML FEATURE DATA
    # --------------------------------------------------------

    st.subheader(
        "🤖 ML Window Feature Dataset"
    )


    st.dataframe(

        window_df.head(100).round(4),

        use_container_width=True

    )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "ℹ️ About Project":

    st.header(
        "ℹ️ About the Project"
    )


    st.markdown(
        """
        ## 🚗 Smart Chassis Vibration Analysis System

        This software prototype demonstrates a vibration-based
        approach for monitoring simulated chassis behavior.

        ---

        ## 🎯 Objective

        The objective of the project is to analyze vibration
        signals and identify patterns that differ from a
        simulated normal chassis baseline.

        ---

        ## ⚙️ Processing Pipeline

        ### 1. Vibration Data

        Simulated vibration signals are generated for
        different chassis conditions.

        ### 2. Time-Domain Analysis

        Statistical features such as:

        - Mean
        - Standard deviation
        - RMS
        - Peak value

        are extracted from the vibration signal.

        ### 3. Frequency-Domain Analysis

        Fast Fourier Transform (FFT) is used to identify
        dominant frequency components.

        ### 4. Window Segmentation

        The vibration signal is divided into smaller
        windows for machine-learning analysis.

        ### 5. Feature Extraction

        Each window contains features including:

        - Mean
        - Standard deviation
        - RMS
        - Peak
        - Dominant frequency
        - Dominant magnitude

        ### 6. Machine Learning

        Isolation Forest is used to identify vibration
        windows that differ from the simulated normal
        baseline.

        ### 7. Risk Assessment

        The proportion of anomalous windows is converted
        into a prototype status:

        - NORMAL
        - WARNING
        - CRITICAL

        ### 8. Safety Telltale

        The application displays a safety-oriented indication
        based on the calculated prototype risk level.

        ---

        ## 🧠 Technologies Used

        - Python
        - NumPy
        - Pandas
        - SciPy
        - Scikit-learn
        - Matplotlib
        - Streamlit

        ---

        ## 📁 Project Structure

        ```
        Smart-Chassis-Vibration/
        │
        ├── app.py
        │
        ├── data/
        │   ├── vibration_dataset.csv
        │   ├── ml_window_features.csv
        │   ├── ml_condition_summary.csv
        │   └── evaluation/
        │
        ├── models/
        │   ├── window_anomaly_model.pkl
        │   └── window_feature_scaler.pkl
        │
        └── src/
            ├── data_generator.py
            ├── vibration_analysis.py
            ├── frequency_analysis.py
            ├── ml_training.py
            └── model_evaluation.py
        ```

        ---

        ## ⚠️ Important Limitation

        This is a software simulation prototype using
        simulated vibration data.

        The system demonstrates the concept of vibration
        analysis and anomaly detection.

        The safety indication is not a certified vehicle
        structural diagnostic system and should not be used
        as a real-world vehicle safety decision system.
        """
    )


# ============================================================
# CONDITION COMPARISON
# ============================================================

st.divider()


st.subheader(
    "📊 All Condition Comparison"
)


comparison = summary_df.copy()


comparison = comparison.rename(

    columns={

        "Anomaly_Percentage":
            "Anomaly_Rate_%",

        "Mean_Dominant_Frequency":
            "Mean_Frequency_Hz"

    }

)


if "Anomaly_Rate_%" in comparison.columns:

    comparison[
        "Anomaly_Rate_%"
    ] = comparison[
        "Anomaly_Rate_%"
    ].round(2)


if "Mean_RMS" in comparison.columns:

    comparison[
        "Mean_RMS"
    ] = comparison[
        "Mean_RMS"
    ].round(4)


if "Mean_Peak" in comparison.columns:

    comparison[
        "Mean_Peak"
    ] = comparison[
        "Mean_Peak"
    ].round(4)


if "Mean_Frequency_Hz" in comparison.columns:

    comparison[
        "Mean_Frequency_Hz"
    ] = comparison[
        "Mean_Frequency_Hz"
    ].round(2)


st.dataframe(

    comparison,

    use_container_width=True

)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Smart Chassis Vibration Analysis System | "
    "Software Simulation Prototype"
)