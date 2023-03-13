# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""
import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy import integrate

# Define a method for solving a initial value problem to solve the Riccati ODE
def solve_riccati_ode(H, M, D, C, R, n, tgrid):

    # Define the associated Riccati ODE
    def riccati_ode(t, S_s):
       S_s = S_s.reshape((n, n))
       dS_ds = -2*np.matmul(np.transpose(H), S_s) + np.matmul(np.matmul(S_s, M), np.linalg.inv(D))*np.matmul(M, S_s) - C
       return dS_ds.flatten()
   
    # Use a backward integration method, starting from the terminal condition S(T)=R
    S = np.zeros((tgrid.size, n, n))
    S[-1 ,: ,:] = R
    for i in range(tgrid.size-2, -1, -1):
       sol = solve_ivp(riccati_ode, [tgrid[i], tgrid[-1]], S[i+1 ,: ,:].flatten())
       S[i ,: ,:] = np.reshape(sol.y[:, -1], (n, n)) # Transform it back to a 2x2 matrix and store the result into S matrix
       S_torch = torch.Tensor(S) # Transform the array into a torch tensor
    return S_torch

# Define a method for solving the value function
def value_function(tgrid, x, sigma, S, batch_size):
    first_term = torch.bmm(torch.bmm(x.transpose(1, 2), S), x).squeeze(1)
    
    # Use the trapezoidal rule to calculate the integral term
      # Calculate the integrand term
    sigma_ = sigma.repeat(batch_size, 1, 1) # Repeat sigma to make it having the same dimension with S
    product = torch.bmm(torch.bmm(sigma_, sigma_.transpose(1, 2)), S)
    diagonal_elements = torch.diagonal(product, dim1=1, dim2=2) 
    integrand = torch.sum(diagonal_elements,dim=1).unsqueeze(1)
      # Define the dr
    dr = torch.tensor((tgrid[-1]-tgrid[0])/(tgrid.size-1))
      # Apply the trapezoidal rule
    second_term = torch.zeros(batch_size, 1)
    trapez_first_term = torch.zeros(batch_size, 1)
    trapez_first_term[0] = integrand[-1]
    trapez_second_term = torch.zeros(batch_size,1)
    # For the uniform Δt, we can modify the trapezoidal rule by spliting it into two parts
    for i in range(0, batch_size, 1):
      trapez_first_term[i] = torch.sum(integrand[i+1:tgrid.size-1])
      trapez_second_term[i] = (integrand[-1]+integrand[i])/2
      second_term[i] = (trapez_first_term[i] + trapez_second_term[i])*dr
    v = first_term + second_term
    return v

# Define a method for showing the optimal Markov control function
def control_function(x, D, M, S, batch_size):
    D_ = D.repeat(batch_size, 1, 1)
    MT_ = M.repeat(batch_size, 1, 1).transpose( 1, 2)
    constant_product = -torch.bmm(D_, MT_)
    a = torch.bmm(constant_product, torch.bmm(S,x))
    a_squeeze = a.squeeze(2)
    return a_squeeze