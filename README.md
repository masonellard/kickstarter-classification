# Predicting Kickstarter Project Success

I analyzed 20,000 projects from 2014 to 2018 on the popular crowdfunding platform: Kickstarter. I built classification models using linear regression, K-nearest neighbors, random forest, and xgboost in order to predict whether a project would meet its funding goal by the deadline.

## File Descriptions

### [kickstarter.py](https://github.com/masonellard/kickstarter-classification/blob/main/kickstarter.py)
Python file that cleans the dataset, conducts feature engineering (including the creation of a time series feature using ARIMA), and classifies project successes and failures using linear regression, KNN, random forest and xgboost. Also includes some visualizations of features and model performance.

### [Data](https://github.com/masonellard/kickstarter-classification/blob/main/data)
Contains all data files (csv and pickle) used in kickstarter.py.

## Results
The xgboost classification model with an ARIMA component seemed to outperform all other models with an ROC AUC of .98. 

Key performance metrics:
* **Precision:** .93
* **Recall:** .94
* **Accuracy:** .92

Key features:
* **Backers:** Number of contributers to a project. Most prominent feature used in final model.
* **Real Goal:** Inflation-adjusted goal in USD. Second most prominent feature used in final model.
* **ARIMA:** ARIMA (order of (1, 1, 0)) prediction of ratio of successes to failures on any given day. Meant to capture any variance that may be attributed to economic fluctuations. Least prominent feature used in final model - though there was relatively little economic volatility during the time period observed. Could be considered safety net that allows the model to adjust to major economic shocks.
