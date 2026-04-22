# AeroPredict MVP (Phase 3)

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
