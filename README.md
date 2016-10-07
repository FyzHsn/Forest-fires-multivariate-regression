Forest-fires-multivariate-regression
====================================

Introduction
------------

In this repository we model the damage area due to forest fires as a function of the following variables:   
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
13. area - the burned area of the forest in ha: 0.00 to 1090.84 (this output variable is very skewed towards 0.0 - logarithm transform might be very useful)  

The data comes from the UCI website: 
    http://archive.ics.uci.edu/ml/datasets/Forest+Fires
    
The citation to this data set:
P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. 
In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, 
Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, 
Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9.

Exploratory Data Analysis
-------------------------

This is a list of figures showing the intercorrelation between all the features. Though this figure looks confusing due to its size, it is a very useful tool.   
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/ExploratoryFigures.png?raw=true)  

Histogram of burn area variable:   
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/untransformed_area.png?raw=true)  

Transformed area variable is more convenient to deal with:   
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/transformed_area.png?raw=true)  

Here it seems there isn't a strong correlation between any of the variables and forest fires:   
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/correlation_heat_map.png?raw=True)
