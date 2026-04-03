# CodeML 2025 - Natural Gas Price Prediction

## Project Overview

This repository contains the solution for the **CodeML 2025** hackathon challenge, focused on forecasting **Henry Hub Natural Gas Spot Prices**. The project explores two distinct modeling approaches to predict daily price variations and absolute values:

1.  **Deep Learning Approach (`main.ipynb`)**: A feature-rich Bidirectional LSTM model leveraging external datasets (weather, news sentiment, gas storage).
2.  **Hybrid Ensemble Approach (`main2.ipynb`)**: A robust baseline combining Gradient Boosting (LightGBM) and a Feedforward Neural Network using technical indicators.

## Approaches & Methodologies

### 1. LSTM Deep Learning Pipeline (`main.ipynb`)
This notebook implements a sophisticated time-series forecasting pipeline using PyTorch.

*   **Model Architecture**: 
    *   **Bidirectional LSTM**: 2 layers, 64 hidden units, with Batch Normalization and Dropout (0.3).
    *   **Input Sequence**: 30-day historical window.
*   **Feature Engineering**:
    *   **Temporal**: Cyclic encoding (sine/cosine) for months and days to capture seasonality.
    *   **Meteorological**: Heating Degree Days (HDD) and Cooling Degree Days (CDD) fetched via `meteostat` for key hubs (Chicago, New York, Dallas).
    *   **Sentiment**: GDELT news sentiment scores (`DecayedEmotionalScore`).
    *   **Fundamental**: National Gas Storage levels.
    *   **Leakage Prevention**: rigorous feature shifting to ensure no future data is used for prediction.
*   **Prediction Strategy**: 
    *   Predicts daily *price variation* rather than absolute price to improve stationarity.
    *   Reconstructs final prices using an $N-1$ actual price correction strategy for maximum accuracy.

### 2. Hybrid Ensemble Baseline (`main2.ipynb`)
This notebook provides a fast, robust alternative using technical analysis.

*   **Models**:
    *   **LightGBM**: A gradient boosting framework trained on tabular features.
    *   **Simple Neural Network**: A 3-layer feedforward network (Dense 64 -> 32 -> 1).
    *   **Ensemble**: Weighted average (70% LightGBM + 30% NN).
*   **Feature Engineering**:
    *   **Lags**: 1, 2, 3, 7, and 14-day lag features.
    *   **Rolling Statistics**: 7-day and 14-day moving averages.
    *   **Date Embeddings**: Day of week, month, year.

## Dataset Description

The solution aggregates data from multiple sources:

*   **`train_henry_hub_natural_gas_spot_price_daily 1.csv`**: Historical daily spot prices (Training data).
*   **`donnees_recentes.csv` / `recent.csv`**: Recent price data used for validation and sequence generation.
*   **`weather_features.csv`**: Precomputed HDD/CDD metrics derived from `meteostat` data.
*   **`codeml_gas_storage.csv`**: Weekly natural gas storage inventory levels (Total Lower 48).
*   **`gdelt_bigquery.csv`**: News sentiment signals extracted from the GDELT project.
*   **`submission.csv`**: Final predictions formatted for the competition.
*   **`test-template.csv`**: Template file for submission structure.

## Key Files

| File | Description |
|------|-------------|
| `main.ipynb` | **Primary Solution**: LSTM pipeline, data fetching, feature engineering, and training. |
| `main2.ipynb` | **Alternative Solution**: LightGBM + NN hybrid model. |
| `natural_gas_lstm_model.pth` | Saved PyTorch model weights for the LSTM. |
| `pytorch_lstm_evaluation.png` | Performance visualization (Loss curves & Predicted vs Actual). |

## Setup & Usage

### Prerequisites
The project requires Python 3.8+ and the following libraries:

```bash
pip install pandas numpy torch scikit-learn lightgbm yfinance meteostat matplotlib tqdm
```

### Running the Code

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd GasPricePrediction
    ```

2.  **Generate Weather/External Data** (Optional):
    *   `main.ipynb` includes logic to fetch fresh weather data using `meteostat` if `weather_features.csv` is missing.

3.  **Train & Predict**:
    *   Run **`main.ipynb`** to train the LSTM model and generate the primary `submission.csv`.
    *   (Optional) Run **`main2.ipynb`** to test the hybrid baseline performance.

## Results
The LSTM model evaluates nicely on validation data:
*   **Training RMSE**: ~0.28
*   **Validation RMSE**: ~0.16
*   Visual results can be seen in `pytorch_lstm_evaluation.png`.

---
*CodeML 2025 Submission*