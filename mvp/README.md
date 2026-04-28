# AeroPredict MVP (Phase 3)
A machine learning model for predicting turbofan engine Remaining Useful Life (RUL).


## Overview
AeroPredict is a simple predictive maintenance model that estimates Remaining Useful Life (RUL) of turbofan engines using NASA C-MAPSS sensor data.  
This Phase 3 MVP builds on earlier phases by training a machine learning model and generating RUL predictions along with basic uncertainty estimates and evaluation metrics.

## MVP Definition 
A user can input turbofan engine sensor data, and the system returns:
- Predicted Remaining Useful Life (RUL)
- Confidence interval (uncertainty estimate)
- Model evaluation metrics (MAE, RMSE)

## Objective
The goal of this MVP is to:
- Load and process engine sensor data  
- Compute Remaining Useful Life (RUL)  
- Train a machine learning model  
- Predict RUL for test engines  
- Evaluate model performance  

## System Design 
Pipeline: 
1. Load raw sensor data
2. Preprocess data (scaling, cleaning, RUL computation)
3. Train regression model
4. Predict RUL
5. Output predictions, uncertainty, and evaluation metrics
   
## Data
This project uses the NASA C-MAPSS FD001 dataset, which contains simulated turbofan engine degradation data.

The dataset includes:
- `train_FD001.txt` → full engine life cycles  
- `test_FD001.txt` → partial engine life cycles  
- `RUL_FD001.txt` → true remaining life for test engines  

⚠️ These files are not included in the repository due to size.

To download them, run:

```bash
wget -O train_FD001.txt https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/train_FD001.txt
wget -O test_FD001.txt https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/test_FD001.txt
wget -O RUL_FD001.txt https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/RUL_FD001.txt
```
### Data Processing 
- Target variable: Remaining Useful Life (RUL)
- Features: Engine sensor readings + operating conditions
- Train/Test split: Provided by dataset (train/test files)

Preprocessing:
- Standardization using StandardScaler
- RUL calculated from engine cycles

## Model

We implemented:
- Baseline: Linear Regression (OLS)
- Improved Model: LASSO Regression (with cross-validation)

Why LASSO:
- Reduces multicollinearity
- Performs feature selection by shrinking coefficients

## How to Run

```bash
python aeropredict_mvp.py
```
Make sure the dataset is inside:

```text
mvp/data/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
```

## Output

- Predicted Remaining Useful Life (RUL)
- Confidence interval (uncertainty bounds)
- Evaluation metrics (MAE, RMSE)

## Evaluation

Metrics used:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

Interpretation:
- Current errors are relatively high, indicating difficulty predicting long-term RUL
- Likely due to:
  - Lack of time-series modeling
  - Sensor noise
  - Limited feature engineering

## Example Output 

```text
MAE: 75.52
RMSE: 86.20
80% interval coverage: 0.00%

Engine 1: Predicted RUL = 0.00 cycles, Interval = [0.00, 0.00], True RUL = 112.00

## Limitations

- Uncertainty estimation failed (0% interval coverage)
  → Confidence intervals are not properly calibrated
- Model does not capture temporal dependencies in engine degradation
- Sensitive to noise and outliers in sensor data

## Next Steps

- Implement time-series models (LSTM / RNN)
- Improve uncertainty estimation (Bayesian approaches)
- Add feature engineering for sensor trends
- Build a simple UI (Streamlit)
