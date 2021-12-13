# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:12:54 2021

@author: Juliana
"""

from abc import ABC, abstractmethod

class Layer(ABC):
    
    def __init__(self):
        self.input=None
        self.output=None
    
    @abstractmethod
    def forward(input):
        raise NotImplementedError
        
    
    @abstractmethod
    def backward(output_error, learning_rate):
        raise NotImplementedError
    
    
    
    
class Dense(Layer):
    
    def __init__(self, input_size, output_size):
        '''Fully Connected layer'''
        self.weights=np.random.rand(input_size, output_size)-0.5
        self.bias=np.random.rand(1,output_size)
        
    def setWeights(self,weights,bias):
        
        if(weights.shape != self.weights.shape):
            raise ValueError(f'Shapes mismatch {weights.shape} and {self.weights.shape}')
        if(bias.shape != self.bias.shape):
            raise ValueError(f'Shapes mismatch {bias.shape} and {self.bias.shape}')
        self.weights=weights
        self.bias=bias
        
    def forward(self, input):
        self.input = input
        np.dot(self.input, weights)+bias
        
        
    def backward(self, output_error,learning_rate):
        raise NotImplementedError
        
        
class Activation(Layer):
    def __init__(self, activation):
        self.activation= activation
        
        
        
    def forward(self,input_data):
        self.input =input_data
        self.output = self.activation(self.input)        
        return self.output
    
    def backward(self, output_error,learning_rates):
        raise NotImplementedError
    
    
class RN(Model):
    
    def __init__(self, epochs=100,lr=0.001, verbose=True):
        self.epochs = epochs
        self.lr =lr
        self.verbose = verbose()
        
        self.layers=[]
        self.loss=mse
        
        
    def add(self,layer):
        '''Add a layer to the notebook'''
        self.layers.append(layer)
        
    def fit(self,dataset):
        raise NotImplementedError
    
    def predict(self,input_data):
        assert self.is_fitted, 'Model must be fit before predict'
        output = input_data
        for layer in layers:
            output = layer.forward(output)
        return output
    
    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit before predict'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)