# Smart Chassis Vibration Analysis

Python and Streamlit prototype for vibration analysis, anomaly detection, and chassis risk assessment using simulated data.

## Structure

```text
app.py                 Streamlit dashboard
src/                   Data generation and analysis scripts
data/                  Generated datasets and evaluation results
models/                Saved ML models and scalers
requirements.txt       Python dependencies
PROJECT_REPORT.md      Technical report
```

## Setup

Use Python 3.10 or newer from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Pipeline

```powershell
python src/data_generator.py
python src/vibration_analysis.py
python src/frequency_analysis.py
python src/anomaly_detection.py
python src/ml_training.py
python src/risk_assessment.py
python src/model_evaluation.py
```

## Start the Dashboard

```powershell
streamlit run app.py
```
