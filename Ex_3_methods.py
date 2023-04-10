# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:35:03 2023

@author: 97896
"""
import torch
import torch.nn as nn
from Ex_1_methods import LQR
from Ex_2_methods import Net_DGM

# Define the methods for geting gradients and Hessian matrix
def get_gradient(output, x):
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad

def get_hess(grad, x):
    hess_diag = []
    for d in range(x.shape[1]):
        v = grad[:,d].view(-1,1)
        grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
        hess_diag.append(grad2)    
    hess_diag = torch.stack(hess_diag,1)
    return hess_diag


# Deep Galerkin method for solving PDE
class PDE_DGM_Bellman(nn.Module):

    def __init__(self, d: int, hidden_dim: int, alpha:float, H:float, M: float, C: float, D:float, R:float, sigma:float, T:float):

        super().__init__()
        self.d = d
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        
        self.net_dgm = Net_DGM(d, hidden_dim, activation='Tanh')
        self.loss_list = []
        self.error_list =[]
        
    def fit(self, epoch: int, t_batch, x_batch):
        batch_size = t_batch.shape[0]
        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        for i in range(epoch):
            optimizer.zero_grad()
            
            # Equation
            x = x_batch
            t = t_batch
            u = self.net_dgm(t, x)
            dx = get_gradient(u,x).unsqueeze(2)
            dt = get_gradient(u, t)
            dxx = get_hess(dx, x)
            
            product = self.sigma@ self.sigma.transpose(0,1) @  dxx
            diagonal_elements = torch.diagonal(product, dim1=1, dim2=2).unsqueeze(2)
            trace = torch.sum(diagonal_elements,dim=1)
            target_functional = torch.zeros_like(u)
            f = dx.transpose(1,2)@self.H@x.unsqueeze(2) + dx.transpose(1,2)@self.M@self.alpha + x.unsqueeze(2).transpose(1,2) @ self.C @x.unsqueeze(2) + self.alpha.transpose(0,1)@self.D@self.alpha
            pde = dt + 0.5*trace + f
            MSE_functional = loss_fn(pde, target_functional)
            
            # Terminal condtion
            x_terminal = x_batch
            t_terminal = torch.ones(batch_size, 1) * self.T
            u_terminal = self.net_dgm(t_terminal, x_terminal)
            terminal_condition = u_terminal - x_terminal.unsqueeze(2).transpose(1,2)@self.R@x_terminal.unsqueeze(2)
            target_terminal = torch.zeros(batch_size,1,dtype=float)
            MSE_terminal = loss_fn(terminal_condition, target_terminal)

            loss = MSE_terminal + MSE_functional
            loss.backward(retain_graph = True)
            optimizer.step()
            self.loss_list.append(loss)
            if i % 10 == 0:
                print(f"Epoch {i}: Loss = {loss.item()}")
                
    def get_loss(self):
            return self.loss_list
    
    # Get the NN solution 
    def get_solution(self, t_batch, x_batch):
            return self.net_dgm(t_batch.unsqueeze(1),x_batch.squeeze(2))
        
    # Error aginst MC solution   
    def aginst_MC(self,interval, t_batch, x_batch):
        lqr = LQR(self.H, self.M, self.sigma, self.C, self.D, self.R, self.T, 2)
        MC_test = lqr.MC_solution_afix(1000,1000,t_batch,x_batch)
        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        for i in range(interval):
            optimizer.zero_grad()
            error = loss_fn(self.get_solution(t_batch, x_batch),MC_test)
            error.backward()
            optimizer.step()
            self.error_list.append(error)
            if i % 10 == 0:
                print(f"Epoch {i}: Error = {error.item()}")
        return MC_test
       
    def get_error(self):           
        return self.error_list