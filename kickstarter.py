#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:26:14 2020

@author: mason
"""

import pandas as pd 
import numpy as np
import seaborn as sns
from datetime import date
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from itertools import combinations
from tqdm import tqdm
from xgboost import XGBClassifier
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


##### DATA CLEANING AND ENGINEERING #####

df = pd.read_csv('/home/mason/Metis/project-3/kickstarter_data_update.csv')
df.columns
df.dtypes

df.isna().sum()

df.dropna(how='any', inplace=True)

df.describe()
df['state'].value_counts()

# Lets drop any rows not classified as 'Failed' or 'Successful'
df.drop(df[(df.state == 'canceled') | (df.state == 'live') | (df.state == 'suspended')].index, inplace=True)

df['state'].value_counts()

# Converting target to boolean
df.loc[df[df.state == 'successful'].index, 'state'] = 1
df.loc[df[df.state == 'failed'].index, 'state'] = 0

# Convert main_category to dummies
df['main_category'].unique()
df = pd.concat([df, pd.get_dummies(df['main_category'])], axis=1)
df.drop(columns=['main_category'], inplace=True)

# Engineering 'days' feature that displays number of days a campaign was open
df.loc[df.index, 'deadline'] = pd.to_datetime(df['deadline'])
df.loc[df.index, 'launched'] = pd.to_datetime(df['launched'])
df['days'] =(df['deadline'].dt.date - df['launched'].dt.date).dt.days

len(df['category'].unique())

# Dropping some columns
df.drop(columns=['id', 'name', 'category', 'currency', 'country', 'usd_pledged', 'usd_pledged_real'], inplace=True)
df.drop(columns=['pledged'], inplace=True)

# Engineering an ARIMA feature that contains ARIMA predictions (rolling, 1 day ahead) of daily success ratios
df_time = df.copy('deep')

df_time['successful'] = (df['state']==True).astype(int)
df_time['failed'] = (df['state']==False).astype(int)
df_time = df_time[['deadline', 'successful', 'failed']].groupby(df_time.deadline.dt.date).sum()

# Success ratio appears stationary
plt.plot(df_time.index, df_time['successful']/(df_time['successful']+df_time['failed']))

df_time['success_ratio'] = df_time['successful']/(df_time['successful']+df_time['failed'])
df_time.drop(columns=['successful', 'failed'], inplace=True)

# Lets plot the ACF and PACF
plot_acf(df_time)
plot_pacf(df_time)

# ACF and PACF indicate AR(1), but we may need to difference
stat_test = adfuller(df_time)
print('p_value:', stat_test[1])

# Lets try some different ARIMA models
model = ARIMA(df_time, order=(1, 0, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model = ARIMA(df_time, order=(2, 0, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model = ARIMA(df_time, order=(0, 0, 1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model = ARIMA(df_time, order=(0, 0, 2))
model_fit = model.fit(disp=0)
print(model_fit.summary())

model = ARIMA(df_time, order=(1, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Looks like a (1, 1, 0) works well, lets get a one day ahead forecast
model = ARIMA(df_time, order=(1, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.forecast()[0])

# Now we need rolling, 1-day-ahead predictions to include as a feature in the model
arima_pred = []
for index in range(10, len(df_time)): # starting at 10th day so that our first forecast has enough degrees of freedom
    model = ARIMA(df_time.iloc[:-len(df_time)+index], order=(1, 1, 0))
    model_fit = model.fit(disp=0)
    arima_pred.append(model_fit.forecast()[0])

# Lets plot our predictions against actual observations    
plt.plot(df_time.iloc[1010:].index, arima_pred[1000:])
plt.plot(df_time.iloc[1010:])


arima_series = pd.Series(data = arima_pred, index = df_time.iloc[10:].index)
arima_series.head()
type(arima_series.index[0])

arima_df = pd.DataFrame(arima_series)
arima_df.reset_index(level=0, inplace=True)

arima_df.deadline = pd.to_datetime(arima_df.deadline)
arima_df.columns = ['deadline', 'arima_p']
arima_df.arima_p = arima_df.arima_p.astype(float)

df = pd.merge(df, arima_df, right_on='deadline', left_on = 'deadline', how='inner')


df.head()
df.dtypes

df, df_test, y, y_test = df.loc[df[df.deadline.dt.date < date(2017, 1, 1)].index], \
                         df.loc[df[df.deadline.dt.date >= date(2017, 1, 1)].index], \
                         df.loc[df[df.deadline.dt.date < date(2017, 1, 1)].index, 'state'], \
                         df.loc[df[df.deadline.dt.date >= date(2017, 1, 1)].index, 'state']

y.drop(df[df.deadline.dt.date < date(2014, 1, 11)].index, inplace = True)
df.drop(df[df.deadline.dt.date < date(2014, 1, 11)].index, inplace = True)

df_deadline = pd.Series(data=df.deadline, index=df.index)
df_test_deadline = pd.Series(data=df_test.deadline, index=df_test.index)

df.drop(columns=['deadline', 'launched', 'state'], inplace=True)
df_test.drop(columns=['deadline', 'launched', 'state'], inplace=True)

y=y.astype('int')
y_test=y_test.astype('int')

# BASE MODEL

reg = LogisticRegression()
fit = reg.fit(df[['backers']], y)
predict = reg.predict(df_test[['backers']])
log_confusion = confusion_matrix(y_test, predict)
log_confusion
print('Logistic Precision:', log_confusion[0][0]/(log_confusion[0][0]+log_confusion[0][1]))
print('Logistic Recall:', log_confusion[0][0]/(log_confusion[1][0]+log_confusion[0][0]))
print('Logistic Accuracy:', (log_confusion[0][0]+log_confusion[1][1])/len(predict))

cv_scores = cross_validate(reg, df[['backers']], y, cv=10, scoring=('roc_auc'), verbose = 0, n_jobs = -1)
a = np.mean(cv_scores['test_score'])
print("Outcomes from Base Classification Model:")
print("Average Test AUC:", a.round(5))


reg = LogisticRegression()
fit = reg.fit(df[['goal']], y)
predict = reg.predict(df_test[['goal']])
log_confusion = confusion_matrix(y_test, predict)
log_confusion
print('Logistic Precision:', log_confusion[0][0]/(log_confusion[0][0]+log_confusion[0][1]))
print('Logistic Recall:', log_confusion[0][0]/(log_confusion[1][0]+log_confusion[0][0]))
print('Logistic Accuracy:', (log_confusion[0][0]+log_confusion[1][1])/len(predict))


# KNN

KNN_parameters = {'n_neighbors': np.arange(1,21,1)}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid = KNN_parameters, cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
grid_search.fit(df, y)
print("Outcomes from the Best KNN Regression Model:")
print("Minimum Average Area Under Curve:", grid_search.best_score_.round(3))
print("The optimal n:", grid_search.best_params_['n_neighbors'])
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(df, y)
knn_confusion = confusion_matrix(y_test, knn.predict(df_test))
knn_confusion
print('KNN Precision:', knn_confusion[0][0]/(knn_confusion[0][0]+knn_confusion[0][1]))
print('KNN Recall:', knn_confusion[0][0]/(knn_confusion[1][0]+knn_confusion[0][0]))
print('KNN Accuracy:', (knn_confusion[0][0]+knn_confusion[1][1])/len(y_test))

# Logistic

# generating list of combos of all features except for first 6
x_combos = []
for n in range(1, len(df.iloc[:,3:].columns)+1):
   combos = combinations(df.iloc[:, 3:].columns, n)
   x_combos.extend(combos)


auc = {}    
for n in tqdm((range(0, len(x_combos)))):
       combo_list = list(x_combos[n]) + list(df.iloc[:, :3].columns)
       x = df[combo_list]
       ols = LogisticRegression()
       cv_scores = cross_validate(ols, x, y, cv=10, scoring=('roc_auc'), n_jobs=-1)
       auc[str(combo_list)] = np.mean(cv_scores['test_score'])
print("Outcomes from the Best Logistic Regression Model:")
max_auc = max(auc.values())
print("Maximum Average AUC:", max_auc.round(5))
for possibles, a in auc.items():
    if a == max_auc:
        print("The Combination of Variables:", possibles)
        ols_features = eval(possibles)
        
reg = LogisticRegression()
fit = reg.fit(df[ols_features], y)
predict = reg.predict(df_test[ols_features])
log_confusion = confusion_matrix(y_test, predict)
log_confusion
print('Logistic Precision:', log_confusion[0][0]/(log_confusion[0][0]+log_confusion[0][1]))
print('Logistic Recall:', log_confusion[0][0]/(log_confusion[1][0]+log_confusion[0][0]))
print('Logistic Accuracy:', (log_confusion[0][0]+log_confusion[1][1])/len(predict))

# Random Forest 


rf = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1100, num = 11)]
rf_parameters = {'n_estimators': n_estimators}
grid_search = GridSearchCV(estimator=rf, param_grid = rf_parameters, cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
grid_search.fit(df, y)
print("Outcomes from the Best Random Forest Model:")
print("Minimum Average Area Under Curve:", grid_search.best_score_.round(3))
print("The optimal n_estimators:", grid_search.best_params_['n_estimators'])

rf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'])
rf.fit(df, y)
rf_predict = rf.predict(df_test)
rf_confusion = confusion_matrix(y_test, rf_predict)
rf_confusion
print('RF Precision:', rf_confusion[0][0]/(rf_confusion[0][0]+rf_confusion[0][1]))
print('RF Recall:', rf_confusion[0][0]/(rf_confusion[1][0]+rf_confusion[0][0]))
print('RF Accuracy:', (rf_confusion[0][0]+rf_confusion[1][1])/len(rf_predict))

# XGBoost

learning_rate = [round(x,2) for x in np.linspace(start = .01, stop = .6, num = 60)]
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
max_depth = range(3,10,1)
child_weight = range(1,6,2)
gamma = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,2]
subsample = [.6, .7, .8, .9, 1]
col_sample = [.6, .7, .8, .9, 1]

# Tuning the learning_rate:
xgb_tune = XGBClassifier(n_estimators = 100,max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'learning_rate':learning_rate},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_learning_rate = xgb_grid.best_params_['learning_rate']
print("Best learning_rate:", best_learning_rate)

# Tuning the n_estimators:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'n_estimators': n_estimators},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_n = xgb_grid.best_params_['n_estimators']
print("Best n_estimators:", best_n)

# Tuning max_depth and min_child_weight:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'max_depth': max_depth, 'min_child_weight': child_weight},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_depth = xgb_grid.best_params_['max_depth']
best_weight = xgb_grid.best_params_['min_child_weight']
print("Best max_depth:", best_depth)
print("Best min_child_weight:", best_weight)

# Tuning gamma:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = .8, colsample_bytree = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'gamma': gamma},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_gamma = xgb_grid.best_params_['gamma']
print("Best gamma:", best_gamma)

# Tuning subsample and colsample_bytree:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

# Rigorously tuning subsample and colsample_bytree:
subsample = [best_subsample-.02, best_subsample - .01, best_subsample, best_subsample +.01, best_subsample + .02]
col_sample = [best_col_sample-.02, best_col_sample - .01, best_col_sample, best_col_sample+.01, best_col_sample+ .02]

xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(df,y)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

xgb = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = best_subsample, colsample_bytree = best_col_sample, gamma = best_gamma, n_jobs = -1)
cv_scores = cross_validate(xgb, df, y, cv=10, scoring=('roc_auc'), verbose = 0, n_jobs = -1)
a = np.mean(cv_scores['test_score'])
print("Outcomes from the Best XGBoost Classification Model:")
print("Average Test AUC:", a.round(5))


xgb.fit(df, y)
importances = list(xgb.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature,importance in zip(df.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance:{}'.format(*pair)) for pair in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
sorted_importances = [importance[1] for importance in feature_importances]
features = []
for i in range(0, len(sorted_features)):
    if sorted_importances[i] != 0:
        features.append(sorted_features[i])
        
print(features)

X = df[features]
cv_scores = cross_validate(xgb, X, y, cv=10, scoring=('roc_auc'), verbose = 0, n_jobs = -1)
a = np.mean(cv_scores['test_score'])
print("Outcomes from the Best XGBoost Classification Model:")
print("Average Test AUC:", a.round(5))

learning_rate = [round(x,2) for x in np.linspace(start = .01, stop = .6, num = 60)]
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
max_depth = range(3,10,1)
child_weight = range(1,6,2)
gamma = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,2]
subsample = [.6, .7, .8, .9, 1]
col_sample = [.6, .7, .8, .9, 1]

# Tuning learning_rate:
xgb_tune = XGBClassifier(n_estimators = 100,max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'learning_rate':learning_rate},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_learning_rate = xgb_grid.best_params_['learning_rate']
print("Best learning_rate:", best_learning_rate)

# Tuning n_estimators:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'n_estimators': n_estimators},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_n = xgb_grid.best_params_['n_estimators']
print("Best n_estimators:", best_n)

# Tuning max_depth and min_child_weight
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'max_depth': max_depth, 'min_child_weight': child_weight},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_depth = xgb_grid.best_params_['max_depth']
best_weight = xgb_grid.best_params_['min_child_weight']
print("Best max_depth:", best_depth)
print("Best min_child_weight:", best_weight)

# Tuning gamma:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = .8, colsample_bytree = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'gamma': gamma},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_gamma = xgb_grid.best_params_['gamma']
print("Best gamma:", best_gamma)

# Tuning subsample and colsample_bytree:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

# Rigorously tuning subsample and colsample_bytree:
subsample = [best_subsample-.02, best_subsample - .01, best_subsample]
col_sample = [best_col_sample-.02, best_col_sample - .01, best_col_sample]

xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=10, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X,y)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

xgb = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = best_subsample, colsample_bytree = best_col_sample, gamma = best_gamma, n_jobs = -1)
cv_scores = cross_validate(xgb, X, y, cv=10, scoring=('roc_auc'), verbose = 0, n_jobs = -1)
a = np.mean(cv_scores['test_score'])
print("Outcomes from the Best XGBoost Classification Model:")
print("Average Test AUC:", a.round(5))

xgb.fit(X,y)
xgb_pred = xgb.predict(df_test[features])
xgb_confusion = confusion_matrix(y_test, xgb_pred)
xgb_confusion
print('XGBoost Precision:', xgb_confusion[0][0]/(xgb_confusion[0][0]+xgb_confusion[0][1]))
print('XGBoost Recall:', xgb_confusion[0][0]/(xgb_confusion[1][0]+xgb_confusion[0][0]))
print('XGBoost Accuracy:', (xgb_confusion[0][0]+xgb_confusion[1][1])/len(xgb_pred))

import pickle

with open('kickstarter_xgb_arima.pickle', 'wb') as to_write:
    pickle.dump(xgb, to_write)
    
with open('kickstarter_xgb_arima_features.pickle', 'wb') as to_write:
    pickle.dump(X.columns, to_write)

with open('kickstarter_xgb_arima.pickle','rb') as read_file:
    xgb = pickle.load(read_file)
    
with open('kickstarter_xgb_arima_features.pickle','rb') as read_file:
    features = pickle.load(read_file)

# obtaining feature importances for xgboost model    
importances = list(xgb.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature,importance in zip(features, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance:{}'.format(*pair)) for pair in feature_importances]

# lets create a bar graph of the feature importances

# first, define the x axis
importance_xaxis = ['Backers',
 'Real Goal',
 'Goal',
 'Film',
 'Games',
 'Tech',
 'Theater',
 'Music',
 'Dance',
 'Design',
 'Comics',
 'Days',
 'Year',
 'Art',
 'Crafts',
 'Photo',
 'Publish',
 'ARIMA',
 'Food']

sns.set_palette('Blues', 1, .9)
sns.set_style('white')
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(importance_xaxis, [x[1] for x in feature_importances])
ax.set_yticklabels(np.round(np.arange(0, .5, .05), 2),Fontsize=18)
ax.set_xticklabels(importance_xaxis, fontsize=11)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# lets graph model performance for the base model, KNN, random forest, and xgboost

# first, we create the data
perf_data = [[.9844, 'XGBoost', 'AUC'],[.981, 'Random Forest', 'AUC'], [.923, 'KNN', 'AUC'], [.9293, 'Base', 'AUC'], [.9327, 'XGBoost', 'Precision'], [.9299, 'Random Forest', 'Precision'],
             [.9171, 'KNN', 'Precision'], [.9592, 'Base', 'Precision'], [.9363, 'XGBoost', 'Recall'], [.9352, 'Random Forest', 'Recall'], [.8022, 'KNN', 'Recall'], [.7586, 'Base', 'Recall'], [.9234, 'XGBoost', 'Accuracy'], 
             [.9212, 'Random Forest', 'Accuracy'], [.8190, 'KNN', 'Accuracy'], [.7974, 'Base', 'Accuracy']]

perf_df = pd.DataFrame(perf_data, columns=['Values', 'Model', 'Metric' ])

plt.figure(figsize=(30, 10))
sns.set_palette('cubehelix')
sns.set_style('ticks')
sns.catplot(x='Model', y='Values',hue='Metric', kind='bar', data = perf_df, ci='sd', alpha=.7)
plt.yticks(np.arange(0, 1.1, .1), fontsize=16)
plt.xticks(fontsize=11)
plt.xlabel(' ')
plt.ylabel(' ')