# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""
import torch
import numpy as np
from scipy.integrate import solve_ivp
from Ex_1_1_matrices import t0, T
# Define a method for solving a initial value problem to solve the Riccati ODE
def solve_riccati_ode(H, M, D, C, R, n, N, tgrid):

    # Define the associated Riccati ODE
    def forward_riccati_ode(s, F_s, H, M, D, C):
       F_s = F_s.reshape((n, n))
       dF = 2*np.transpose(H)@F_s - F_s@M@np.linalg.inv(D)@M@F_s + C
       return dF.flatten()
   
    # Use a backward induction method, starting from the terminal condition S(T)=R
    F = np.zeros((N+1, n, n))
    S = np.zeros((N+1, n, n))
    F[0] = R
    sol = solve_ivp(forward_riccati_ode, (tgrid[0], tgrid[N]), F[0].flatten(), args=(H,M,D,C), t_eval=tgrid)
    F = sol.y.T.reshape(-1,2,2)
    for i in range(0,N):
      S[N-i] = F[i]
    S[0] = F[-1]
    S = torch.tensor(S)
    return S

# Define a method for solving the value function
def value_function(N, x, sigma, S):
    first_term = (x.transpose(1, 2)@S@x).squeeze(1)
    tgrid = np.linspace(t0,T,N+1)
    # Calculate the integrand term
    product = sigma @ sigma.transpose(0, 1)@S
    diagonal_elements = torch.diagonal(product, dim1=1, dim2=2) 
    integrand = torch.sum(diagonal_elements,dim=1).unsqueeze(1)
    second_term = torch.zeros(N+1, 1)
      # Define the dr
    dr = torch.tensor((tgrid[-1]-tgrid[0])/(tgrid.size-1))
    #def trapez_uniform(dr,integrand):
      #integral = torch.zeros(N+1, 1)
     # trapez_first_term = torch.zeros(N+1, 1)
      #trapez_first_term[0] = integrand[-1]
      #trapez_second_term = torch.zeros(N+1,1)   
      # For the uniform Δt, we can modify the trapezoidal rule by spliting it into two parts
      #for i in range(0, N+1):
        #trapez_first_term[i] = torch.sum(integrand[i+1:tgrid.size-1])
        #trapez_second_term[i] = (integrand[-1]+integrand[i])/2
        #integral[i] = (trapez_first_term[i] + trapez_second_term[i])*dr
    for i in range (0,N):
          second_term[i] = torch.sum(integrand[i:N+1],dim=0)*dr 
    #return second_term
    #second_term = trapez_uniform(dr, integrand)
    v = first_term + second_term
    return v

# Define a method for showing the optimal Markov control function
def control_function(x, D, M, S, batch_size):
    D = torch.tensor(D)
    M = torch.tensor(M)
    MT = M.transpose(0, 1)
    constant_product = -D@MT
    a = constant_product@S@x
    a_squeeze = a.squeeze(2)
    return a_squeeze