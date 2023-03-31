# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:03:39 2023

@author: 97896
"""
import torch
import numpy as np
from Ex_1_1_matrices import (n, t0, T)
#from Ex_1_1_methods import value_function
# Construct a Monte Carlo simulation
# P is the number of samples
# N is the number of time steps
# MSE represents the mean squared error
#N = 1000
def MC_checks(x,sigma,S,v_real,H, M, D, C, P, N):
    H = torch.tensor(H)
    M = torch.tensor(M)
    D = torch.tensor(D)
    C = torch.tensor(C)
    #sigma_r = sigma.repeat(N+1, 1, 1)# Repeat sigma to make it having the same dimension with S
    product = sigma @ sigma.transpose(0, 1) @ S
    diagonal_elements = torch.diagonal(product, dim1=1, dim2=2) 
    integrand = torch.sum(diagonal_elements,dim=1).unsqueeze(1)
    #integrand = integrand.repeat(P,1,1)
    #S = S.repeat(P,1,1,1)
    #sigma_r = sigma_r.repeat(P,1,1,1)
    dt = (T-t0)/N
    v = torch.zeros(size=(P,N+1,1))
    x_simulated = torch.zeros(size=(P,N+1,n,1),dtype=float)
    x_simulated[:,0] = x[0]
    W = torch.normal(mean=0.0, std=1.0,size = (P,N+1,n,1),dtype=float)
    for i in range(0,N):
            x_simulated[:,i+1] = x_simulated[:,i] + dt*(H@x_simulated[:,i]-M@D@M.transpose(0,1)@S[i]@x_simulated[:,i])+sigma@(W[:,i+1]-W[:,i])*np.sqrt(dt)
            v[:,i]=(x_simulated[:,i].transpose(1,2)@S[i]@x_simulated[:,i]).squeeze(2)+torch.sum(integrand[i:N+1],dim=0)*dt          
    v_mean = torch.sum(v,dim=0)/P
    v_mean = torch.sum(v_mean,dim=0)/N
    v_real = torch.sum(v_real,dim=0)/N
    error = v_mean-v_real
    return error     