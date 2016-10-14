# -*- coding: utf-8 -*-
"""
In this script, I take the forest fires dataset and create a new feature 
variable which stores the incidence of forest fire or no forest fire. Then, I
wish to fit a SVM to the dataset. Lastly, tune parameters for optimal 
performance.

Correction: I have decided to classify forest fires according to the following
range of areas: = 0, > 0 & < 1, > 1 & < 10, > 10 & < 100 and lastly > 100. 
Why have I done this? To me the area burnt down sems very chaotic, so I am
interested in predicting order of magnitudes rather than exact area.

Author: Faiyaz Hasan
Date: October 8, 2016
"""
#############################
# 0. PACKAGES AND LOAD DATA #
#############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

file_location = r'C:\Users\Windows\Dropbox\AllStuff\Forest_fire_damage\Data\forestfires.csv'
df = pd.read_csv(file_location)

#########################
# 1. DATA PREPROCESSING #
#########################

# delete day of week information - don't expect it to be relevant
del df['day']

# extract feature and target variables from dataframe
X = df.iloc[:, 0:11].values    
y= df.iloc[:, 11].values

# split training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=0)

# standardize numeric feature variables of dataset
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train[:, 3:11])
X_test_std = stdsc.transform(X_test[:, 3:11])

############################################
# 2. MODEL AND TEST TRAINING DATA WITH SVM #
############################################

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'C': param_range,
               'kernel': ['linear']},
               {'C': param_range,
                'gamma': param_range,
                'kernel': ['rbf']}]

# optimize parameters via GridSearchCV
svm = SVC(random_state=0)
gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10)
gs = gs.fit(X_train_std, y_train)
print(gs.best_score_)
print(gs.best_params_)

svm = SVC(kernel='rbf', gamma=0.001, C=1000.0, random_state=0)
svm.fit(X_train_std, y_train)
print('SVM Training score: ', svm.score(X_train_std, y_train)*100)


# feature importance with random forest
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0)
forest.fit(X_train_std, y_train)
print('Feature importance: ', forest.feature_importances_)









