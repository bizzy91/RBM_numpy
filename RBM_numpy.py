#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:01:00 2018

@author: bizzy
"""

# For numerical calculation
import numpy as np
import numpy.random as rd
# For Drawing graphs
import matplotlib.pyplot as plt
# For importing MNIST data
import input_data
# For sigmoid function
from scipy.special import expit
import csv    

# Import MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX = mnist.train.images

n_v = 784
n_h = 500
batchsize = 50
R = 0.1

# Initialize weights and biases
w = rd.normal(0, 0.01, (n_v, n_h))
a = np.zeros(n_v)
b = np.zeros(n_h)

def Sigmoid(x):
    return expit(x)

for epoch in range(100):
    choice = rd.randint(55000, size=(batchsize,))
    
    v = np.zeros((batchsize,n_v))
    h = np.zeros((batchsize,n_h))
    vh  = np.zeros((batchsize, n_v, n_h))
    vh1 = np.zeros((batchsize, n_v, n_h))
    
    for i, j in zip(range(batchsize), choice):
        v[i] = trX[j]
    
    
    h = Sigmoid(np.dot(v, w) + b)
    for j in range(batchsize):
        vh[j] = np.dot(v[j].reshape(n_v,1), h[j].reshape(1,n_h))
    
    v1 = Sigmoid(np.dot(h, w.T) + a) 
    h1 = Sigmoid(np.dot(v1, w) + b)
    
    for k in range(batchsize):
        vh1[k] = np.dot(v1[k].reshape(n_v,1), h1[k].reshape(1,n_h))  
    
    w += R * np.mean((vh-vh1), 0)
    a += R * np.mean(v - v1, 0)
    b += R * np.mean(h - h1, 0)
    if epoch%100 == 0:
        print(np.mean((v-v1)*(v-v1)), "epoch : ", epoch)

# Test plot
r = rd.randint(55000)
plt.subplot(4,4,1); plt.imshow(trX[r].reshape(28,28));

old_v = trX[r]
for i in range(15):
    new_h = Sigmoid(np.dot(old_v.reshape(1, n_v), w) + b)
    new_v = Sigmoid(np.dot(new_h, w.T) + a)
    plt.subplot(4,4,2+i); plt.imshow(new_v.reshape(28,28));
    old_v = new_v
    
# Save weights and biases
# Visible bias
save_data = open("a1.csv", "w"); wr = csv.writer(save_data); # a1
wr.writerow(a)
save_data.close()


# Hidden bias
save_data = open("b1.csv", "w"); wr = csv.writer(save_data); # b1
wr.writerow(b)
save_data.close()


# Weight matrix
save_data = open("w1.csv", "w"); wr = csv.writer(save_data); # W1
for i in range(n_v):
    wr.writerow(w[i])
save_data.close()
