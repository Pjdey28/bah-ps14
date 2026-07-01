# Energetic Particle Radiation Forecast

[[Streamlit App]](https://solar-storm-forecast.streamlit.app/)

This project provides an end-to-end machine learning system for forecasting energetic particle radiation (solar storms). It leverages an MLOps approach to process satellite data, train a sophisticated ensemble of models, and evaluate their performance across multiple time horizons. The results are explorable through an interactive Streamlit dashboard.

## Key Features

- **Ensemble Machine Learning**: Utilizes a Ridge Stacking ensemble combining Random Forest, LightGBM, XGBoost, LSTM, and GRU models for robust and accurate predictions.
- **Multi-Horizon Forecasting**: Generates forecasts for 30-minute, 6-hour, and 12-hour horizons.
- **Interactive Dashboard**: A comprehensive [Streamlit dashboard](https://solar-storm-forecast.streamlit.app/) to explore predictions, compare model performance, and analyze feature importance.
- **Model Explainability**: Integrates SHAP (SHapley Additive exPlanations) to provide insights into how features influence model predictions.
- **Reproducible Pipeline**: The entire pipeline, from data ingestion to model deployment, is designed to be reproducible.
- **Real-time Monitoring**: A conceptual real-time monitoring system with a React frontend and a Python backend designed for live data ingestion and forecasting.

## Technology Stack

| Category                      | Technologies                                                                                             |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Machine Learning**          | `Scikit-learn`, `PyTorch`, `LightGBM`, `XGBoost`, `SHAP`                                                   |
| **Data Handling & Core**      | `Pandas`, `NumPy`, `SciPy`, `xarray`, `netCDF4`, `cdflib`, `PyArrow`                                       |
| **Application & Visualization** | `Streamlit`, `Plotly`, `Matplotlib`, `React`                                                             |
| **Real-time Backend (Conceptual)** | `Flask`, `Socket.IO`                                                                                     |

## Project Architecture

The project features a dual architecture: a batch MLOps pipeline for model training and evaluation, and a real-time pipeline for live forecasting.

```
[ Raw GOES/WIND Data ] 
          |
    ( Ingestion ) --> [ Processed Parquet Files ]
                              |
                     ( Feature Engineering )
                              |
          +-------------------+-------------------+

          |                                       |
  [ L: Tree Models ]                    [ L: Sequence Models ]
  (RF, XGB, LGBM)                        (LSTM, GRU)

          |                                       |
          +-------------------+-------------------+
                              |
                 [ L: Stacking Meta-Model ]
                     (Ridge Regression)
                              |
               ( SHAP Evaluation & Artifacts )
                              |
          +-------------------+-------------------+

          |                                       |
 [ Streamlit Dashboard ]               [ Kafka Live Stream ]
                                                 |
                                     [ Live React Dashboard ]

```

## Technical Deep Dive

### 1. Data Preprocessing
The pipeline begins by ingesting raw satellite data from GOES (NetCDF format) and WIND (CDF format).
- **Loading**: Scripts in `src/data` handle the loading of these complex file formats into memory.
- **Cleaning**: Data is cleaned to handle missing values, outliers, and inconsistencies.
- **Time Alignment**: Datasets from different sources are resampled and aligned to a common time index to ensure chronological consistency.
- **Merging**: The aligned datasets are merged into a single, unified DataFrame.
- **Storage**: The final processed data is saved in the efficient Parquet format under `data/processed/`, making it ready for the feature engineering stage.

### 2. Feature Engineering & Selection
- **Feature Creation**: New features are derived from the time-series data, such as rolling averages, lagged values, and interaction terms, to capture temporal dynamics and complex relationships. This is handled by scripts in `src/features`.
- **Feature Selection**: SHAP-based feature importance is used to analyze the contribution of each feature. Less important or redundant features can be pruned to reduce model complexity and improve performance. The results of this analysis are visible in the Streamlit dashboard.

### 3. Model Architecture
The project employs a two-level stacking ensemble to combine the strengths of different types of models:
- **Level 0 (Base Models)**:
  - **Tree-Based Models**: `RandomForest`, `XGBoost`, and `LightGBM` are used for their ability to capture non-linear relationships in tabular feature sets.
  - **Sequence-Based Models**: `LSTM` and `GRU` networks are used to model the temporal dependencies inherent in time-series satellite data.
- **Level 1 (Meta-Model)**:
  - A `Ridge` regression model is used as the meta-learner. It takes the predictions from all Level 0 models as its input features and generates the final, blended forecast. This approach helps to smooth out the errors of individual models and produce a more robust prediction.

### 4. Training, Inference, and Evaluation
- **Training**: The `src/training` module orchestrates the training of all base models and the final ensemble. Trained model objects and scalers are serialized and saved as artifacts.
- **Inference**: The `run_model_comparison.py` script performs batch inference on the test set, generating predictions for all models across the 30-minute, 6-hour, and 12-hour horizons.
- **Evaluation**: The `src/evaluation` module calculates key performance metrics (`RMSE`, `MAE`, `R2`) for each model and horizon. It generates:
  - **Comparison Plots**: Bar charts comparing metrics across all models.
  - **Detailed Plots**: Per-model plots including forecast vs. actual, scatter plots of predictions, and residual analysis.
  - **Data Artifacts**: CSV files containing raw metrics, summary tables, and prediction samples for deep-dive analysis in the dashboard.

### 5. Logging
Standard Python `logging` is configured throughout the `src` pipeline. It provides crucial visibility into the execution flow by:
- Tracking the progress of long-running tasks like data processing and model training.
- Recording key information, such as data shapes, file paths, and model parameters.
- Capturing warnings and errors for easier debugging and maintenance.

## How to Run

### Prerequisites

- Python 3.9+
- Node.js & npm
- Git

### Running the Project

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

**2. Run the Streamlit Dashboard**

The dashboard is the primary interface for exploring the pre-generated results of the ML pipeline.

```bash
cd dashboard
pip install -r requirements.txt
streamlit run Home.py
```

Navigate to `http://localhost:8501` in your browser.

> **Note**: The ML pipeline scripts (`src/evaluation/run_model_comparison.py`, `src/evaluation/shap_analysis.py`) are designed to run in an environment with access to the raw GOES and WIND datasets. As these datasets are not included in the repository, running the pipeline from scratch is not directly supported. The project is intended to be explored via the dashboard using the existing artifacts.

## Architectural Improvement: Live Monitoring System

As an extension, this project includes a demonstration of a real-time monitoring system. This system is composed of a frontend application that visualizes a live data stream. The live-backend and live-monitor will ingest GRASP data.
### Future Data Integration: ISRO GRASP/GSAT
A key technical improvement for this system will be the integration of real-time data from **ISRO's GRASP/GSAT** (Geostationary Satellite) payloads. This will allow for testing, validation, and operational forecasting using a new, high-quality data source.
### Data Streaming with Kafka
To handle the high-volume, real-time data from sources like GRASP, the architecture incorporates **Apache Kafka** as a streaming platform. Kafka provides a scalable, fault-tolerant message bus to decouple data producers (satellites) from data consumers (the inference service).

```
[ Data Source ]              [ Messaging ]              [ Backend ]               [ Frontend ]
ISRO GRASP/GSAT  ----->  Kafka Producer  ----->  Prediction Engine  ----->  React Live Monitor
   Data Stream                |                  (Model Loading)          (Socket.IO Stream)
                              v                         |                         |
                      Kafka Topic: solar-data <---------+                 Real-time Charts

```

### Live Monitor (Frontend)

The `/live-monitor` directory contains a React-based frontend application that provides a live-updating dashboard. It is designed to visualize a stream of solar activity data and corresponding forecasts.

**Key Features**:
- Built with React and Vite for a modern, fast development experience.
- Uses Recharts for creating interactive and responsive time-series charts.
- Includes a mock data stream within the frontend, allowing it to run standalone without a live backend connection. This is perfect for demonstration and UI development.

**To Run the Frontend Demo:**

```bash
cd live-monitor
npm install
npm run dev
```
The React application will start (typically on `http://localhost:5173`) and display a dashboard with replaying mock data.

### Live Backend (Conceptual)

The `/live-backend` directory contains the code for a Flask-SocketIO server. Conceptually, this backend would:
1.  **Ingest Data**: Connect to a real-time data source, such as a GRASP data feed or a message queue like Apache Kafka.
2.  **Run Inference**: Load the trained ensemble model from the `/artifacts` directory.
3.  **Generate Forecasts**: As new data packets arrive, preprocess them and generate live forecasts.
4.  **Stream Results**: Push the incoming data and the corresponding forecasts to all connected frontend clients via Socket.IO.

This component is provided as a blueprint and is not required to run the live monitor demo.

## Project Structure

- **/src**: Core Python source code for the ML pipeline.
  - `data/`: Scripts for data ingestion and preprocessing (e.g., from GOES, WIND).
  - `features/`: Scripts for feature engineering and selection.
  - `models/`: Definitions of machine learning models (e.g., LSTM, GRU, Ensemble).
  - `training/`: Logic for training models and running the main pipeline.
  - `evaluation/`: Scripts for model evaluation, comparison, and SHAP analysis.
  - `config.py`: Central configuration for paths, parameters, and constants.
- **/dashboard**: Streamlit application for model analysis and visualization.
- **/artifacts**: Output directory for all generated files.
  - `models/`: Serialized trained models, scalers, and ensemble objects.
  - `evaluation/`: CSV files with metrics, prediction samples, and PNG plots.
  - `shap/`: SHAP value plots and feature importance data.
- **/data**: Directory for raw and processed datasets.
- **/live-backend**: Flask and Socket.IO server for the real-time monitoring system.
- **/live-monitor**: React-based frontend for the live dashboard.
- **/notebook**: Jupyter notebook for exploratory data analysis (EDA).
