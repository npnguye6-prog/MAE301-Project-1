# AeroPredict MVP Report

## Executive Summary
AeroPredict is a predictive maintenance MVP that estimates Remaining Useful Life (RUL) for turbofan engines using NASA C-MAPSS sensor data. The Phase 3 MVP successfully loads the dataset, computes RUL labels, trains a machine learning model, and generates RUL predictions with uncertainty intervals. The current MVP demonstrates a complete end-to-end pipeline, but model accuracy and uncertainty quality are still limited.

## User and Use Case
The target users are mechanical or aerospace engineers and maintenance planners who need a quick estimate of engine health and remaining life. The intended use case is to help support maintenance decisions by turning raw sensor data into predicted RUL values and uncertainty bounds.

## System Design
The system takes engine sensor data as input, processes the last available cycle for each engine, and uses a Random Forest regressor to predict RUL. The workflow is:
1. Load NASA C-MAPSS data
2. Compute RUL labels for training engines
3. Extract engine-level features from the latest cycle
4. Train the model
5. Predict RUL for test engines
6. Estimate uncertainty using the spread of tree predictions

## Data
The MVP uses the NASA C-MAPSS FD001 turbofan engine degradation dataset. The training file contains full run-to-failure trajectories, and the test file contains partial engine histories with provided RUL labels. Features include operating settings and multiple sensor channels.

## Model
The final MVP uses a Random Forest regressor as a simple supervised learning baseline. The model was chosen because it is easy to train, handles nonlinear relationships better than linear regression, and works well for a first MVP. Uncertainty was approximated using the spread of predictions across the individual trees in the forest.

## Evaluation
The MVP was evaluated on the FD001 test set using standard regression metrics.

- MAE: 75.52
- RMSE: 86.20
- 80% interval coverage: 0.00%

These results show that the pipeline runs successfully, but prediction quality is still weak. The uncertainty intervals also did not capture the true values well, so the current uncertainty method is not yet reliable.

## Limitations and Risks
The main limitation is that the current model uses only simple engine-level features from the last cycle, which likely removes important degradation trends across time. The uncertainty estimation is also very basic and not well calibrated. Another limitation is that no advanced feature engineering, normalization, or sequence-based modeling was used in this MVP.

## Next Steps
With more time, the next improvements would be:
- use rolling or window-based sensor features
- normalize and select the most useful sensors
- compare multiple models such as ridge regression, gradient boosting, or XGBoost
- improve uncertainty estimation
- add plots and a cleaner demo interface
- generate an automatic markdown summary/report for each engine case

## Conclusion
The Phase 3 MVP successfully demonstrates an end-to-end AeroPredict workflow using real NASA engine data. Although the current model accuracy is limited, the project shows a working predictive maintenance prototype and identifies clear directions for improvement.
