# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""
import torch
import numpy as np
from scipy.integrate import solve_ivp

class LQR:
  # Initialize the matrices
  def __init__(self, H, M, sigma, C, D, R, T, n):
        self.H = H
        self.M = M
        self.sigma = sigma
        self.C = C
        self.D = D
        self.R = R
        self.T = T
        self.n = n
        
  # Define a method for solving a initial value problem to solve the Riccati ODE          
  def solve_riccati_ode(self, tgrid):
    # Define the associated Riccati ODE
    def forward_riccati_ode(s, F_s):
       F_s = F_s.reshape((self.n, self.n))
       dF = np.transpose(self.H)@F_s + F_s@ - F_s@self.M@np.linalg.inv(self.D)@self.M@F_s + self.C
       return dF.flatten()
   
    # Get the solution of the Riccati ODE
    F = np.zeros((tgrid.size, self.n, self.n))
    S = np.zeros((tgrid.size, self.n, self.n))
    F[0] = self.R
    sol = solve_ivp(forward_riccati_ode, (tgrid[0], tgrid[-1]), F[0].flatten(), t_eval=tgrid)
    F = sol.y.T.reshape(-1,2,2)
    S = np.flip(F, axis=0)
    S = torch.tensor(S.copy())
    return S

  def value_function(self, t_batch, x_batch):
      batch_size = t_batch.shape[0]
      v = torch.zeros((batch_size,1),dtype=float)
      for i in range(batch_size):
          t = t_batch[i]
          x = x_batch[i]
          N = 100
          teval = np.linspace(t, self.T, N+1)
          S = self.solve_riccati_ode(teval)
          product = self.sigma @ self.sigma.transpose(0, 1)@ S
          diagonal_elements = torch.diagonal(product, dim1=1, dim2=2) 
          integrand = torch.sum(diagonal_elements,dim=1).unsqueeze(1)
          dr = (self.T-t)/N
          v[i] = x.transpose(0,1)@S[0]@x + torch.sum(integrand,dim=0)*dr #Use trapzoid rule
      return v
  
  # Define a method for showing the optimal Markov control function
  def control_function(self, t_batch, x_batch):
     Dinv = torch.tensor(np.linalg.inv(self.D),dtype=float)
     M = torch.tensor(self.M)
     batch_size = t_batch.shape[0]
     a = torch.zeros((batch_size,2,1),dtype=float)
     for i in range(batch_size):
        t = t_batch[i]
        x = x_batch[i]
        N = 100
        teval = np.linspace(t, self.T, N+1)
        S = self.solve_riccati_ode(teval)
        a[i] = -Dinv@M.transpose(0,1)@S[0]@x
     a = a.squeeze(2)
     return a
  
  # Define the Monte Carlo simulation and the measurement of errors
  def MC_checks(self, N_MC, N_timesteps, t_batch, x_batch):
      H = torch.tensor(self.H,dtype=float)
      M = torch.tensor(self.M,dtype=float)
      D = torch.tensor(self.D,dtype=float)
      Dinv = torch.tensor(np.linalg.inv(self.D),dtype=float)
      C = torch.tensor(self.C,dtype=float)
      R = torch.tensor(self.R,dtype=float)
      batch_size = t_batch.shape[0]
      dt = torch.tensor((self.T)/N_timesteps, dtype=float)
      v = self.value_function(t_batch, x_batch)
      v_simulated_mean = torch.zeros(batch_size,1)
      S = torch.zeros(batch_size, N_timesteps+1, 2, 2, dtype = float)
      for i in range(batch_size):
         t = t_batch[i]
         teval = np.linspace(t, self.T, N_timesteps+1)
         S[i] = self.solve_riccati_ode(teval)
      v_simulated = 0
      x_next = 0
      x = x_batch.repeat(N_MC,1,1,1).transpose(0,1)
      for j in range(N_timesteps):
             dW = torch.randn(size=(batch_size,N_MC, 1, 1),dtype=float)*torch.sqrt(dt)
             a = -Dinv@M.transpose(0,1)@S[:,j].unsqueeze(1)@x
             x_next = x + (H@x + M@a)*dt + self.sigma@dW
             v_simulated += (x.transpose(2,3)@C@x+a.transpose(2,3)@D@a)*dt  
             x = x_next
      v_simulated += x.transpose(2,3)@R@x
      v_simulated_mean = torch.mean(v_simulated.squeeze(2),dim=1)
      error = torch.mean(torch.square(v_simulated_mean-v),dim=0)
      return error
  
  def MC_solution_afix(self, N_MC, N_timesteps, t_batch, x_batch):
        H = torch.tensor(self.H,dtype=float)
        M = torch.tensor(self.M,dtype=float)
        D = torch.tensor(self.D,dtype=float)
        C = torch.tensor(self.C,dtype=float)
        R = torch.tensor(self.R,dtype=float)
        batch_size = t_batch.shape[0]
        dt = torch.tensor((self.T)/N_timesteps, dtype=float)
        v_simulated_mean = torch.zeros(batch_size,1)
        S = torch.zeros(batch_size, N_timesteps+1, 2, 2, dtype = float)
        for i in range(batch_size):
           t = t_batch[i]
           teval = np.linspace(t, self.T, N_timesteps+1)
           S[i] = self.solve_riccati_ode(teval)
        x = x_batch.repeat(N_MC,1,1,1).transpose(0,1)
        a = torch.tensor([[1],[1]],dtype=float)
        v_simulated = 0
        x_next = 0
        for j in range(N_timesteps):
               dW = torch.randn(size=(batch_size,N_MC, 1, 1),dtype=float)*torch.sqrt(dt)
               x_next = x + (H@x + M@a)*dt + self.sigma@dW
               v_simulated += (x.transpose(2,3)@C@x+a.transpose(0,1)@D@a)*dt  
               x = x_next
        v_simulated += x.transpose(2,3)@R@x
        v_simulated_mean = torch.mean(v_simulated.squeeze(2),dim=1)
        return v_simulated_mean
