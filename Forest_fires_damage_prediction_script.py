# -*- coding: utf-8 -*-
"""
This script contains my analysis of extent of damage during forest fires based
on various parameters within the Montesinho park. 

The data comes from the UCI website: 
    http://archive.ics.uci.edu/ml/datasets/Forest+Fires
    
The relevant citation to this data set is:
P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. 
In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, 
Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, 
Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################
# 0. LOAD DATA #
################
file_location = r'C:\Users\Windows\Dropbox\AllStuff\Forest_fire_damage\Data\forestfires.csv'
df = pd.read_csv(file_location)


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
    1. X - x-axis spatial coordinate with the Montesinho park map: 1 - 9. 
    2. Y - y-axis spatial coordinate with the Montesinho park map: 1 - 9. 
    3. month - month of the year: "jan" to "dec"
    4. day - day of the week: "mon" to "sun"
    5. FFMC - Fine Fuel Moisture Code index from FWI system: 18.7 - 96.20
    6. DMC - Duff moisture code index from the FWI system: 1.1 to 291.3
    7. DC - Drought code index from the FWI system: 7.9 to 860.6
    8. ISI - Initial spread index from the FWI system: 0.0 tp 56.10
    9. temp - temperature in Celsius degrees: 2.2 to 33.30
    10. RH - relative humidity in %: 15.0 to 100
    11. wind - wind speed in km.h: 0.40 to 9.40
    12. rain - outside rain in mm/m2: 0.0 to 6.4
    13. area - the burned area of the forest in ha: 0.00 to 1090.84
    (this output variable is very skewed towards 0.0 - logarithm transform
    might be very useful)
    
"""


################################
# 2. EXPLORATORY DATA ANALYSIS #
################################
sns.set(style='whitegrid', context='notebook')
cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
sns.pairplot(df[cols], size=2.5)
plt.show()

# missing values in data frame - NO missing values
print('*********************')
print(df.isnull().sum())

# what does the target value look like
print('*********************')
print(df['area'].describe())

# histogram of target variable - burnt area
df['area'].hist(bins=200)
plt.savefig('untransformed_area.png')
plt.clf()

# transformation of area to perhaps to deal with skewness
area_transform = np.log(1 + df['area'])
area_transform.hist()
##################################################################
# 3. FIT DATA - REGRESSION ANAYSIS + OUTLIER DETECTION ALGORITHM #
##################################################################

##################################
# 4. PREDICTION ABILITY OF MODEL #
##################################

