# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""

import numpy as np
# Define the time grid
t0 = 0
T = 1
N = 5000
batch_size = N+1 # Define the batch size
tgrid = np.linspace(t0, T, N+1)

# Define the input matrices spicifying the LQR
n = 2 # Define the dimension
#H = np.random.randn(n,n)
#M = np.random.randn(n,n)
H = np.array([[0.5, -0.6],[-0.2, -1.1]])
M = np.array([[ 0.9, -0.9],[0.8, -0.3]])
# Strictly postive definite matrix D
#D = np.abs(np.random.randn(n,n))
#while (D.all() > 0):
  #if (D.all() == 0):
   # D = np.abs(np.random.randn(n,n))
 # else:
   # D = D
   # break
D = np.array([[0.5, 1.0],[0.4, 0.5]])
# Positive definite matrices C and R   
#C = np.abs(np.random.randn(n,n))
#R = np.abs(np.random.randn(n,n))
C = np.array([[1.7, 1.3],[1.2, 0.26]])
R = np.array([[0.6, 0.7],[1.3, 0.8]])

