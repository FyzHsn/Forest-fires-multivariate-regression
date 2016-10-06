# -*- coding: utf-8 -*-
"""
This script contains my analysis of extent of damage during forest fires based
on various parameters. 

The data comes from the UCI website: 
    http://archive.ics.uci.edu/ml/datasets/Forest+Fires
    
The relevant citation to this data set is:
P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. 
In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, 
Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, 
Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9.

"""
import pandas as pd

################
# 0. LOAD DATA #
################
file_location = r'C:\Users\Windows\Dropbox\AllStuff\Forest_fire_damage\Data\forestfires.csv'
df = pd.read_csv(file_location, header=None)


################################
# 1. EXPLORE DATASET STRUCTURE #
################################

# (Row num, Column num) = (517, 13)
print('****************************')
print('(Row #, Column #): ', df.shape)

# Data header
print('****************************')
print(df.head(3))

# Column names
df.columns = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 
              'RH', 'wind', 'rain', 'area']
print('****************************')
print('Column names: ', df.columns)

"""
A brief discussion of the feature vectors:

"""

################################
# 2. EXPLORATORY DATA ANALYSIS #
################################

##################################################################
# 3. FIT DATA - REGRESSION ANAYSIS + OUTLIER DETECTION ALGORITHM #
##################################################################

##################################
# 4. PREDICTION ABILITY OF MODEL #
##################################

