# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:26:01 2021

@author: Juliana
"""

from .model import Model
from ..util.metrics import mse
import numpy as np

class LinearRegression(Model):
    
    def __init__(self, gd=false, epochs=1000,lr=0.001):
        
        super(LinearRegression, self).__init__()
        self.gd=gd
        self.theta=None
        self.epochs=epochs
        self.lr=lr
    
    def fit(self, dataset):
        x, y = dataset.getXy()
        x=np.hstack((np.ones((x.shape[0], 1)), x))
        self.X=X
        self.y=y
        #closed form or GD
        self.train_gd(X,y) if self.gd else self.train_closed(X,y)
        self.is_fitted = True
       
    def train_closed(self, X, y):
        """Uses closed form linear algebra to fit the model.
        theta = inv(XT*X)*XT*y"""
        
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
               
    def train_gd(self, X, y):
        m=X.shape[0]
        n=X.shape[1]
        self.history={}
        self.theta=np.zeros(n)
        for epoch in range(self.epochs):
            grad=1/m * (X.dot(self.theta)-y).dot(X)
            self.theta-= self.lr * grad
            self.history[epoch]=[self.theta[:], self.cost()]
    
    def predict(self,x):
        assert self.is_fittied, 'Model must be fit before predicting'
        _x = np.hstack(([1],x))
        return np.dot(self.theta, _x)
    
    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self._x, y_pred)/2

    
class LinearRegressionReg(LinearRegression):
    
    def __init__(self, gd=False, epochs=1000, lr=0.001, lbd=1):
        super(LinearRegressionReg, self).__init__(gd=gd, epochs=epochs, lr=lr)
        self.lbd=lbd
    
    def train_closed(self, X, y):
        n=X.shape[1]
        identity=np.eye(n)
        identity[0,0] = 0
        self.theta=np.linalg.inv(X.T.dot(X)+self.lbd*identity).dot(X.T).dot(y)
        self.is_fitted = True
    
    def train_gd(self, X, y):
        m=X.shape[0]
        n=X.shape[1]
        self.history={}
        self.theta=np.zeros(n)
        lbds = np.full(m,self.lbd)
        lbds[0]=0
        for epoch in range(self.epochs):
            grad=(X.dot(self.theta)-y).dot(X)
            self.theta -= (self.lr/m)*(lbds+grad)
            self.history[epoch]=[self.theta[:], self.cost()]
    
    def predict(self,x):
        