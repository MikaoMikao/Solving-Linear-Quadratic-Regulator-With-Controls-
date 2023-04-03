# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""

import torch
import numpy as np
from Ex_1_1_methods import LQR
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 1.1 Solving LQR using the Riccati ODE:

# Define the timegrid for solving Riccati ODE    
t0 = 0
T = 1
N = 1000
tgrid = np.linspace(t0, T, N+1)

# Define the input matrices spicifying the LQR
n = 2 # Define the dimension
H = np.array([[1.0, 0.0],[0.0, 1.0]])
M = np.array([[1.0, 0.0],[0.0, 1.0]])
sigma = torch.tensor([[0.05],[0.05]],dtype = float)
D = np.array([[0.1, 0.0],[0.0, 0.1]])
C = np.array([[0.1, 0.0],[0.0, 0.1]])
R = np.array([[1.0, 0.0],[0.0, 1.0]])

# Define the mehods
lqr = LQR(H, M, sigma, C, D, R, T, n)
S_real = lqr.solve_riccati_ode(tgrid)

# Define the batch with batch_size = 100
t_batch = torch.randn(100,dtype = float)
x_batch = torch.randn(100,2,1,dtype = float)

# Get the value function based on the batch
v_real = lqr.value_function(t_batch,x_batch)
print(v_real[0])

# Get the Markov control function based on the batch
a = lqr.control_function(t_batch,x_batch)
print(a[0])

#1.2 LQR MC checks
N_timesteps = 5000
error_list = []
#P_values = {10,50,100,500,1000,5000,10000,50000}
for N_MC in range(25):
   error = torch.log(lqr.MC_checks(int(10**(N_MC/5)), N_timesteps, t_batch, x_batch))
   error_list.append(error)
print(error_list)
#n=range(100,100000,3996)
y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(error_list,'r')
plt.xlabel('Monte Carlo Samples')
plt.ylabel('Mean Square Error')
plt.show()

N_MC = 10**4
MSE_list_time = []
#N_values = {1,10,50,100,500,1000,5000}
for i in range(20):
   MSE = torch.log(lqr.MC_checks(N_MC, int(10**(i/5)), t_batch, x_batch))
   MSE_list_time.append(MSE)
print(MSE_list_time)
n=range(100,10000,495)
y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(n,MSE_list_time,'r')
plt.xlabel('Timesteps')
plt.ylabel('Mean Square Error')
plt.legend()
plt.show()