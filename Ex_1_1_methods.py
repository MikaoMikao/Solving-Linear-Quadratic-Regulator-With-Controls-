# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""
import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm

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
       dF = 2*np.transpose(self.H)@F_s - F_s@self.M@np.linalg.inv(self.D)@self.M@F_s + self.C
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
          v[i] = x.transpose(0,1)@S[0]@x +(integrand[0]+integrand[-1])*0.5*dr + torch.sum(integrand[1:N-1],dim=0)*dr #Use trapzoid rule
      return v
  
  # Define a method for showing the optimal Markov control function
  def control_function(self, t_batch, x_batch):
     D = torch.tensor(self.D)
     M = torch.tensor(self.M)
     batch_size = t_batch.shape[0]
     a = torch.zeros((batch_size,2,1),dtype=float)
     for i in range(batch_size):
        t = t_batch[i]
        x = x_batch[i]
        N = 100
        teval = np.linspace(t, self.T, N+1)
        S = self.solve_riccati_ode(teval)
        a[i] = -D@M.transpose(0,1)@S[0]@x
     a = a.squeeze(2)
     return a
 
  def MC_checks(self, N_MC, N_timesteps, t_batch, x_batch):
      #Define the error measuring function
      H = torch.tensor(self.H)
      M = torch.tensor(self.M)
      D = torch.tensor(self.D)
      C = torch.tensor(self.C)
      R = torch.tensor(self.R)
      #a = self.control_function(t_batch, x_batch)
      v = self.value_function(t_batch, x_batch)
      x_simulated = torch.zeros(size = (N_MC, N_timesteps+1, self.n, 1), dtype=float)
      a_simulated = torch.zeros(size = (N_MC, N_timesteps+1, self.n, 1), dtype=float)
      #v_simulated = torch.zeros(size = (N_MC, 1), dtype=float)
      dt = torch.tensor((self.T)/N_timesteps, dtype=float)
      rvs = norm.rvs(size = (N_MC, N_timesteps+1, 1, 1))
      dW = torch.tensor(rvs)*torch.sqrt(dt)
      #for i in range(batch_size):
      t = t_batch[0]
      x = x_batch[0]
      teval = np.linspace(t,self.T,N_timesteps+1)
      S = self.solve_riccati_ode(teval)
      x_simulated[:,0] = x
      v_simulated = 0
      for i in range(N_timesteps):
          a_simulated[:,i] = -D@M.transpose(0,1)@S[i]@x_simulated[:,i]
          #x_simulated[:,i+1] = x_simulated[:,i] + (H@x_simulated[:,i] - M@D@M.transpose(0,1)@S[i]@x_simulated[:,i])*dt + self.sigma@dW[:,i]
          x_simulated[:,i+1] = x_simulated[:,i] + (H@x_simulated[:,i] + M@a_simulated[:,i])*dt + self.sigma@dW[:,i]
          v_simulated += (x_simulated[:,i].transpose(1,2)@C@x_simulated[:,i]+a_simulated[:,i].transpose(1,2)@D@a_simulated[:,i])*dt   
      v_simulated += x_simulated[:,-1].transpose(1,2)@R@x_simulated[:,-1]
      v_simulated_mean = torch.sum(v_simulated.squeeze(2),dim=0)/N_MC
      ABS = torch.abs(v_simulated_mean-v[0])
      return ABS