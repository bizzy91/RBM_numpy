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
from RBM_class import RBM

# Import MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX = mnist.train.images




R = RBM(784,500,30,100,0.1,trX)
R.CD()
R.Save_data("a","b","w")

w = R.w
a = R.a
b = R.b


n_v = 784
n_h = 500

r = rd.randint(55000)
plt.subplot(4,4,1); plt.imshow(trX[r].reshape(28,28));

old_v = trX[r]
for i in range(15):
    new_h = R.Sigmoid(np.dot(old_v.reshape(1, n_v), w) + b)
    new_v = R.Sigmoid(np.dot(new_h, w.T) + a)
    plt.subplot(4,4,2+i); plt.imshow(new_v.reshape(28,28));
    old_v = new_v