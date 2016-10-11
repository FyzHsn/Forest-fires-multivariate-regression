# -*- coding: utf-8 -*-
"""
In this script, I take the forest fires dataset and create a new feature 
variable which stores the incidence of forest fire or no forest fire. Then, I
wish to fit a SVM to the dataset. Lastly, tune parameters for optimal 
performance.

Author: Faiyaz Hasan
Date: October 11, 2016
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

############################################
# 2. MODEL AND TEST TRAINING DATA WITH SVM #
############################################

svr = SVR()
svr.fit(X_train_std, y_train)
print('SVR Training score: ', svr.score(X_train_std, y_train))








