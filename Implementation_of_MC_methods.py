# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""

import torch
import numpy as np
from Ex_1_methods import LQR
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

# Apply the mehods to get the solution of the Riccati ODE
lqr = LQR(H, M, sigma, C, D, R, T, n)
S_real = lqr.solve_riccati_ode(tgrid)

# Define the batch with batch_size = 25
batch_size = 25
t_batch = torch.rand(batch_size,dtype = float)
x_batch = torch.rand(batch_size,n,1,dtype = float)*0.5

# Get the value function based on the batch
v_real = lqr.value_function(t_batch,x_batch)
print(v_real[0])

# Get the Markov control function based on the batch
a = lqr.control_function(t_batch,x_batch)
print(a[0])

v_s = lqr.MC_solution_afix(1000,100, t_batch, x_batch)
#1.2 LQR MC checks
# Fix the sample size
N_MC = 100000
error_list_time = []
N_values = [1,10,50,100,500,1000,5000]
for N_timesteps in N_values:
   error = torch.log(lqr.MC_checks(N_MC, int(10**(N_timesteps/5)), t_batch, x_batch)) 
   error_list_time.append(error)
print(error_list_time)
n = np.log(N_values)
y_major_locator = MultipleLocator(5)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(n,error_list_time,'r')
plt.xlabel('Log of Timesteps')
plt.ylabel('Log of Mean Square Error')
plt.legend()
plt.show()

# Fix the time steps
N_timesteps = 50
error_list = []
N_MC_values = [10,50,100,500,1000,5000,10000,50000,100000]
for N_MC in N_MC_values:
   error = torch.log(lqr.MC_checks(int(10**(N_MC/5)), N_timesteps, t_batch, x_batch))
   error_list.append(error)
print(error_list)
n = np.log(N_MC_values)
y_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(n,error_list,'b')
plt.xlabel('Log of Monte Carlo Samples')
plt.ylabel('Log of Mean Square Error')
plt.show()
