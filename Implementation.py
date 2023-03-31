# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""

import torch
import numpy as np
from Ex_1_1_matrices import (H, M, D, C, R, n, N, t0, T,tgrid, batch_size)
from Ex_1_1_methods import solve_riccati_ode
from Ex_1_1_methods import value_function
from Ex_1_1_methods import control_function
from MC_checks import MC_checks
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# 1.1 Solving LQR using the Riccati ODE:
S_real = solve_riccati_ode(H, M, D, C, R, n, N, tgrid)
#print(S_real[0])
#print(S_real[-1])
#plt.plot(tgrid,S_real[:,1,1])
#plt.show()
#x = torch.normal(0, 1, size = (N+1,n,1),dtype=float)
linspace1 = np.linspace(-1,1,N+1)
linspace2 = np.linspace(-1,1,N+1)
linspace = [linspace1,linspace2]
x = torch.tensor(linspace).transpose(0,1).unsqueeze(2)
#sigma = torch.normal(0, 1, size = (n,n),dtype = float)
sigma = torch.tensor([[-0.8,0.5],[0.6,0.2]],dtype = float)
v_real = value_function(N, x, sigma, S_real)
#print(v_real[0])
#print(v_real[-1])
#plt.plot(tgrid,v_real)
#plt.show()
a = control_function(x, D, M, S_real, batch_size)
#print(a[0])
#print(a[-1])
#plt.plot(tgrid,torch.sum(a,dim = 1))
#plt.show()
#1.2 LQR MC checks
N = 10000
MSE_list = []
#P_values = {10,50,100,500,1000,5000,10000,50000}
for i in range(25):
   timesteps = np.linspace(t0,T,N+1)
   linspace1 = np.linspace(-1,1,N+1)
   linspace2 = np.linspace(-1,1,N+1)
   linspace = [linspace1,linspace2]
   x = torch.tensor(linspace).transpose(0,1).unsqueeze(2)
   S_real = solve_riccati_ode(H, M, D, C, R, n, N, timesteps)
   v_real = value_function(N, x, sigma, S_real)
   MSE = torch.log(MC_checks(x,sigma,S_real,v_real,H, M, D, C, int(10**(i/5)), N))
   MSE_list.append(MSE)
print(MSE_list)
n=range(100,100000,3996)
y_major_locator = MultipleLocator(0.1)
ax = plt.gca()
plt.plot(n,MSE_list,'r')
plt.xlabel('Monte Carlo Samples')
plt.ylabel('Mean Square Error')
plt.show()

P = 10**5
MSE_list_time = []
#N_values = {1,10,50,100,500,1000,5000}
for i in range(20):
   timesteps = np.linspace(t0,T,int(10**(i/5))+1)
   linspace1 = np.linspace(-1,1,int(10**(i/5))+1)
   linspace2 = np.linspace(-1,1,int(10**(i/5))+1)
   linspace = [linspace1,linspace2]
   x = torch.tensor(linspace).transpose(0,1).unsqueeze(2)
   S_real = solve_riccati_ode(H, M, D, C, R, n, int(10**(i/5)), timesteps)
   v_real = value_function(int(10**(i/5)), x, sigma, S_real)
   MSE = torch.log(MC_checks(x,sigma,S_real,v_real,H, M, D, C, P, int(10**(i/5))))
   MSE_list.append(MSE)
print(MSE_list)
plt.plot(MSE_list)
plt.xlabel('Timesteps')
plt.ylabel('Mean Square Error')
plt.legend()
plt.show()