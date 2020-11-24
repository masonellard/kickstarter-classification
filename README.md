# Predicting Kickstarter Project Success

I analyzed 20,000 projects from 2014 to 2018 on the popular crowdfunding platform: Kickstarter. I built classification models using linear regression, K-nearest neighbors, random forest, and xgboost in order to predict whether a project would meet its funding goal by the deadline.

## File Descriptions

### [kickstarter.py](https://github.com/masonellard/project-3/blob/main/kickstarter.py)
Python file that cleans the dataset, conducts feature engineering (including the creation of a time series feature using ARIMA), and classifies project successes and failures using linear regression, KNN, random forest and xgboost. Also includes some visualizations of features and model performance.

### [Data](https://github.com/masonellard/project-3/blob/main/data)
Contains all data files (csv and pickle) used in kickstarter.py.

## Results
The xgboost classification model with an ARIMA component seemed to outperform all other models with an ROC AUC of .98. 

Key performance metrics:
* **Precision:** .93
* **Recall:** .94
* **Accuracy:** .92
