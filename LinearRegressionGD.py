# -*- coding: utf-8 -*-
"""
Implementation of an OLS linear regression model using the gradient 
descent method.

Author: Faiyaz Hasan
Date Created: October 8, 2016
"""
import numpy as np
class LinearRegressionGD(object):
    """Linear regression analysis.
    
    Parameters
    ----------
    eta : float
        Learning rate between 0.0 and 1.0.
    n_iter : int
        Number of passes over the training dataset.
    
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting. Underscore after a variable name indicates 
        that the variable was not created on instantiation of the object.
    cost_ : list
        Cost function of sample batch and weight vector per epoch.
    errors_ : list
        List of errors after weight update per epoch
        
    """
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit training data according to the adaline algorithm.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training dataset, where n_samples is the number of samples
            and n_features is the number of features.
        y : {array-like}, shape = [n_samples]
            Binary classification of dataset.
                    
        Returns
        -------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        self.errors_ = []
        
        for i in range(self.n_iter):
            err = 0
            # compute errors per epoch
            for j in range(X.shape[0]):
                status = y[j] - self.predict(X[j, ]) 
                err += int(status != 0.0)
                
            # update weights    
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
            
            
            self.errors_.append(err)
            
        return self
        
    def net_input(self, X):
        """Calculate the dot product of the features and the weights. """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return prediction value."""        
        return self.net_input(X)
        
        
        
        
        
        
        
        
        
    

