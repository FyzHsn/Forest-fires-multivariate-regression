# -*- coding: utf-8 -*-
"""
I have decided to classify forest fires according to the following
range of areas: = 0, > 0 & < 1, > 1 & < 10, > 10 & < 100 and lastly > 100. 
Why have I done this? To me the area burnt down sems very chaotic, so I am
interested in predicting order of magnitudes rather than exact area.

Author: Faiyaz Hasan
Date: October 13, 2016
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
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline

file_location = r'C:\Users\Windows\Dropbox\AllStuff\Forest_fire_damage\Data\forestfires.csv'
df = pd.read_csv(file_location)

#########################
# 1. DATA PREPROCESSING #
#########################

# delete day of week information - don't expect it to be relevant
del df['day']

# get dummy variables for nominal features
df = pd.get_dummies(df)

# indices of feature variables
area_index = [i for i in range(0, df.shape[1]) if (df.columns[i] == 'area')]
nonarea_index = [i for i in range(0, df.shape[1]) if (df.columns[i] != 'area')]             

# extract feature and target variables from dataframe
X = df.iloc[:, nonarea_index].values    
y = np.zeros(X.shape[0])
for i in range(0, X.shape[0]):
    if df['area'][i] == 0.0:
        y[i] = '0'
    else:
        y[i] = '1'
        
# split training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=0)

# standardize numeric feature variables of dataset
nonmonth_index = X.shape[1] - 12
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train[:, 0:nonmonth_index])
X_test_std = stdsc.transform(X_test[:, 0:nonmonth_index])

# add month information
X_train_std = np.hstack((X_train_std, X_train[:, nonmonth_index:]))
X_test_std = np.hstack((X_test_std, X_test[:, nonmonth_index:]))

#########################
# 2. FEATURE IMPORTANCE #
#########################

# feature importance with random forests
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0)
forest.fit(X_train_std, y_train)
importances = forest.feature_importances_
feat_labels = df.columns[nonarea_index]
indices = np.argsort(importances)[::-1]

# plot of feature importances
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# feature importance via principal component analysis
cov_mat = np.cov(X_train_std[:, :nonmonth_index].T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(eigen_vals)

# logistic regression before lda
lr = LogisticRegression()
lr.fit(X_train_std, y_train)
print(lr.score(X_train_std, y_train))

# feature importance via linear discriminant analysis - no difference
# really
lda = LDA(n_components=18)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr.fit(X_train_lda, y_train)
print(lr.score(X_train_lda, y_train))

# Straightforward logistic regression seems to do a good job

#########################################
# 3. OPTIMIZE LOGISTIC REGRESSION MODEL #
#########################################

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'C': param_range,
               'penalty': ['l1']},
              {'C': param_range,
               'penalty': ['l2']}]

lr = LogisticRegression(random_state=0)
gs = GridSearchCV(estimator=lr,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10)
gs = gs.fit(X_train_std, y_train)
print(gs.best_score_)
print(gs.best_params_)

# diagnosing bias an dvariance problems with learning curves.
lr = LogisticRegression(penalty='l1', C=10.0)
train_sizes, train_scores, test_scores = \
    learning_curve(estimator=lr,
                   X=X_train_std,
                   y=y_train,
                   train_sizes=np.linspace(0.1, 1.0, 20),
                   cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o',
         markersize=5, label='training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='red', marker='o',
         markersize=5, label='training accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='red')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='top right')
plt.ylim([0.4, 1.0])
plt.show()

#########################
# 4. OPTIMIZE SVM MODEL #
#########################

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'C': param_range,
               'kernel': ['linear']},
               {'C': param_range,
                'kernel': ['poly'],
                'degree': [2]}]

# optimize parameters via GridSearchCV
svm = SVC(random_state=0)
gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10)
gs = gs.fit(X_train_std, y_train)
print(gs.best_score_)
print(gs.best_params_)

# diagnosing bias an variance problems with learning curves.
svm = SVC(kernel='poly', degree=2)
train_sizes, train_scores, test_scores = \
    learning_curve(estimator=svm,
                   X=X_train_std,
                   y=y_train,
                   train_sizes=np.linspace(0.1, 1.0, 20),
                   cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o',
         markersize=5, label='training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='red')
plt.plot(train_sizes, test_mean, color='blue', marker='o',
         markersize=5, label='training accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 1.0])
plt.show()

