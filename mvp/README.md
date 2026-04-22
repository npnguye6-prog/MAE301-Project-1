# AeroPredict MVP (Phase 3)
A machine learning model for predicting turbofan engine Remaining Useful Life (RUL).


## Overview
AeroPredict is a simple predictive maintenance model that estimates Remaining Useful Life (RUL) of turbofan engines using NASA C-MAPSS sensor data.  
This Phase 3 MVP builds on earlier phases by training a machine learning model and generating RUL predictions along with basic uncertainty estimates and evaluation metrics.


## Objective
The goal of this MVP is to:
- Load and process engine sensor data  
- Compute Remaining Useful Life (RUL)  
- Train a machine learning model  
- Predict RUL for test engines  
- Evaluate model performance  


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

## How to Run

```bash
python aeropredict_mvp.py
```
Make sure the dataset is inside:

mvp/data/
train_FD001.txt
test_FD001.txt
RUL_FD001.txt**


## Output

- Predicted Remaining Useful Life (RUL)
- Confidence interval (uncertainty bounds)
- Evaluation metrics (MAE, RMSE)


## Example Output

```text
MAE: 75.52
RMSE: 86.20
80% interval coverage: 0.00%

Engine 1: Predicted RUL = 0.00 cycles, Interval = [0.00, 0.00], True RUL = 112.00
