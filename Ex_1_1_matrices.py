# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""
import torch
import numpy as np

# Define the time grid
t0 = 0
T = 10
N = 100
batch_size = N+1 # Define the batch size
tgrid = np.linspace(t0, T, N+1)

# Define the input matrices spicifying the LQR
n = 2 # Define the dimension
H = torch.normal(0, 1,size=(n, n))
M = torch.normal(0, 1,size=(n, n))
x = torch.normal(0, 1, size = (batch_size,n,1))
sigma = torch.normal(0, 1, size=(n,n))

# Strictly postive definite matrix D
D = abs(torch.normal(0, 1,size=(n, n)))
while (D.all() > 0):
  if (D.all() == 0):
    D = abs(torch.normal(0, 1,size=(n, n)))
  else:
    D = D
    break
    
# Positive definite matrices C and R   
C = abs(torch.normal(0, 1,size=(n, n)))
R = abs(torch.normal(0, 1,size=(n, n)))

