# -*- coding: utf-8 -*-
"""
In this script, I take the forest fires dataset and create a new feature 
variable which stores the incidence of forest fire or no forest fire. Then, I
wish to fit a SVM to the dataset. Lastly, tune parameters for optimal 
performance.

Author: Faiyaz Hasan
Date: October 8, 2016
"""
#############################
# 0. PACKAGES AND LOAD DATA #
#############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_location = r'C:\Users\Windows\Dropbox\AllStuff\Forest_fire_damage\Data\forestfires.csv'
df = pd.read_csv(file_location)

#########################
# 1. DATA PREPROCESSING #
#########################

print('Number of rows with zero burn area: ', np.sum(df['area'] == 0.0))
print('Number of rows with non-zero burn area: ', np.sum(df['area'] != 0.0))
print('Shape of dataset: ', df.shape)

del df['day']
del df['month']

# extract feature and target variables from dataframe
X = df.iloc[:, 0:9].values    
y = df.iloc[:, 10].values

burn_status= np.array(['yes' if i != 0.0 else 'no' for i in df['area']])
burn_status.shape = (df.shape[0], 1)

X = np.hstack((X, burn_status)) 


