# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:32:19 2023

@author: 97896
"""

import torch
from Ex_3_methods import PDE_DGM_Bellman
import matplotlib.pyplot as plt

# Set the bounds of t and x
x_ub = 1
x_lb = -1
t0 = 0
T = 1

# Set matrices
alpha = torch.tensor([[1.0],[1.0]],dtype=float)
H = torch.tensor([[1.0, 0.0],[0.0, 1.0]],dtype=float)
M = torch.tensor([[1.0, 0.0],[0.0, 1.0]],dtype=float)
D = torch.tensor([[0.1, 0.0],[0.0, 0.1]],dtype=float)
C = torch.tensor([[0.1, 0.0],[0.0, 0.1]],dtype=float)
sigma = torch.tensor([[0.05],[0.05]],dtype=float)
R = torch.tensor([[1.0, 0.0],[1.0, 0.0]],dtype=float)

# Train the model with training data
pde = PDE_DGM_Bellman(2, 100, alpha, H, M, C, D, R, sigma, T)
batch_size = 100
x_train = (x_ub-x_lb)*torch.rand(batch_size,2,requires_grad=True,dtype = float)+x_lb
t_train = T*torch.rand(batch_size,1,requires_grad=True,dtype = float)+t0
epochs = 200
pde.fit(epochs,t_train,x_train)
sol = pde.get_solution(t_train.squeeze(1).detach(),x_train.unsqueeze(2).detach())

# Plot the training loss
loss = pde.get_loss()
plt.plot(range(epochs),torch.tensor(loss))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('Training loss')
plt.show()

# Plot the error aginst MC solution
interval = 100
MC_sol = pde.aginst_MC(interval,t_train.squeeze(1).detach(),x_train.unsqueeze(2).detach())
errors = pde.get_error()
plt.plot(range(interval),torch.tensor(errors),'r')
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title('Error Against MC Solution')
plt.show()
