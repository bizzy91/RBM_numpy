#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:01:00 2018

@author: bizzy
"""

# For numerical calculation
import numpy as np
import numpy.random as rd
# For sigmoid function
from scipy.special import expit
import csv    

class RBM:
    def __init__(self, n_visible, n_hidden, n_batch, epochs, learning_rate, training_data):
        self.n_v     = n_visible
        self.n_h     = n_hidden
        self.n_batch = n_batch
        self.epochs  = epochs
        self.R       = learning_rate
        self.trX     = training_data
        
        # Initialize weight and biase
        self.w = rd.normal(0, 0.01, (self.n_v, self.n_h))
        self.a = np.zeros(self.n_v)
        self.b = np.zeros(self.n_h)

        self.v   = np.zeros((self.n_batch, self.n_v))
        self.h   = np.zeros((self.n_batch, self.n_h))
        self.vh  = np.zeros((self.n_batch, self.n_v, self.n_h))
        self.vh1 = np.zeros((self.n_batch, self.n_v, self.n_h))     
        
    # Activation function    
    def Sigmoid(self, x):
        return expit(x)
    
    # Update the weight and bias using CD(Contrastive Divergence) algorithm
    def CD(self):
        for epoch in range(self.epochs):
            print(epoch)
            choice = rd.randint(55000, size=(self.n_batch,))
            
            # Iput batch_data
            for i, j in zip(range(self.n_batch), choice):
                self.v[i] = self.trX[j]
            # Initialize hidden layer
            self.h = self.Sigmoid(np.dot(self.v, self.w) + self.b)
            # Positive gradient, Data
            for j in range(self.n_batch):
                self.vh[j] = np.dot(self.v[j].reshape(self.n_v, 1), self.h[j].reshape(1, self.n_h))
            # CD-1
            self.v1 = self.Sigmoid(np.dot(self.h, self.w.T) + self.a) 
            self.h1 = self.Sigmoid(np.dot(self.v1, self.w) + self.b)
            # Negative gradient, Model
            for k in range(self.n_batch):
                self.vh1[k] = np.dot(self.v1[k].reshape(self.n_v, 1), self.h1[k].reshape(1, self.n_h))  
            
            self.w += self.R * np.mean((self.vh - self.vh1), 0)
            self.a += self.R * np.mean(self.v - self.v1, 0)
            self.b += self.R * np.mean(self.h - self.h1, 0)
            print(epoch)
            if epoch%100 == 0:
                print(np.mean((self.v-self.v1)*(self.v-self.v1)), epoch, "th epoch")
    
    # Each of names has to be string type
    def Save_data(self, a_name, b_name, w_name):
        # Visible bias
        save_data = open(a_name+".csv", "w"); wr = csv.writer(save_data);
        wr.writerow(self.a)
        save_data.close()

        # Hidden bias
        save_data = open(b_name+".csv", "w"); wr = csv.writer(save_data);
        wr.writerow(self.b)
        save_data.close()
                
        # Weight matrix
        save_data = open(w_name+".csv", "w"); wr = csv.writer(save_data);
        for i in range(self.n_v):
            wr.writerow(self.w[i])
        save_data.close()

