Forest-fires-multivariate-regression
====================================

Summary of main results
-----------------------

### Spatial properties of fires
Here are plots showing the spatial results of the forest fires dataset. The next three plots show the number of forest fires, total area burnt and average damaged area per fire in each of the park zones respectively.  
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/FireNumParkZone.png?raw=true)  
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/FireAreaParkZone.png?raw=true)  
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/FireDensityParkZone.png?raw=true)  
* Note that the greatest damaged areas have the most fires with one exception. 
* Even though the areas with the greatest number of fires have the greatest damage done, the damage is mostly due to one huge fire rather than many medium sized fires. In fact most of the firest are extremely negligible.

**Treat with skepticism (but interesting)** This might be evidence for having controlled fires instead of stamping out every tiny fire. Controlled minor fires have the effect of burning out the underbrush and preventing larger extreme forest fires.

### Evidence of power law behaviour
Furthermore, there is some evidence for power law behaviour. 99% of the area burnt is due to 1% of the fires.  

One of the questions would be how do I figure that out? Can you device a metric to measure extremes in area burnt during fires?

Limitations of the dataset: We do not know the extent of human intervention in the size of the fire. Where the fires being put out by forest patrols. Then, that adds an additional level of complication since we do not have information as to detection time and then response time. That adds a hugely complicating factor in making inferences because it is not accounted for in the dataset. 

### Classification of minor and major forest fires via SVM
Since, predicting the burnt area seems to be unviable due to the chaotic nature of fires, I reformulated the problem in terms of classifying minor and major forest fires. Here's the cross validation curve plot:

![](?raw=true)


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

**Note:** It was initially my intention to model forest fires damage area via multivariate regression analysis. However, after conducting exploratory data analysis and looking at the variance of damage area versus the meteorological variables as well as the correlation table, we find that the correlation is extremely small. Furthermore, I expect the burn area to be highly chaotic and hence easy to be fooled into finding a pattern that does not generalize on a different dataset.

Another quick point: It is extremely important to recognize that though independent variables have zero covariance, zero covariance does not imply independence of variables. As an example conside: X and Y = X^2. Cov(X, Y) = 0. 

However, in the EDA (Exploratory data analysis) step, it seemed promising that a classification type of problem could be set up. This seems especially viable when observing that the forest fires taking place in the absence of rain. The classification problem would find the decision boundary of the conditions that lead to a fire. This classification task is feasible with this dataset since almost half the datasets are of instances with 0.0 ha of fire damage. This can easily be engineered into a new feature.

Lastly, for the sake of trying things anyways, I will try a random forest regression approach to fitting the data.

### Objective:
1. Apply SVM algorithm to classify if the burn damage will be non-zero or not.   
2. Apply random forest regression since it is a good tool for non-linear regression.   

Exploratory Data Analysis
-------------------------

This is a list of figures showing the intercorrelation between all the features. Though this figure looks confusing due to its size, it is a very useful tool.   
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/ExploratoryFigures.png?raw=true)  

Histogram of burn area variable:   
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/untransformed_area.png?raw=true)  

Transformed area variable is more convenient to deal with:   
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/transformed_area.png?raw=true)  

The correlation values are pretty much zero. In conjuction with the EDA plots and the absence of obvious patterns, I abandon the idea of multiple regression and instead reformulate this scenario as a classification problem of a forest fire or no forest fire scenario.
![](https://github.com/FyzHsn/Forest-fires-multivariate-regression/blob/master/Figs/correlation_heat_map.png?raw=True)

Binary classification via SVM
-----------------------------

### Step Outline
* Process data frame to favorable form: add new feature.   
* Standardize data set.   
* Test/Training set split of the data set.   
* Apply SVM naively and check accuracy score via cross-validation.   
* Hyper-parameter tuning via grid search.  
* State final conclusions.   







