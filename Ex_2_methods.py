# -*- coding: utf-8 -*-
"""
@author: Zhanhao Zhang
"""

import torch
import torch.nn as nn
from Ex_1_methods import LQR

class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, batch_norm=False):
        super().__init__()
        
        layers = [nn.BatchNorm1d(sizes[0]),] if batch_norm else []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1], dtype=torch.double))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[j+1], affine=True))
            if j<(len(sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True

    def forward(self, x):
        return self.net(x)

class DGM_Layer(nn.Module):
    
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(DGM_Layer, self).__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))
        self.gate_Z = self.layer(dim_x+dim_S, dim_S)
        self.gate_G = self.layer(dim_x+dim_S, dim_S)
        self.gate_R = self.layer(dim_x+dim_S, dim_S)
        self.gate_H = self.layer(dim_x+dim_S, dim_S)
          
    def layer(self, nIn, nOut):
        l = nn.Sequential(nn.Linear(nIn, nOut,dtype=torch.double), self.activation)
        return l
    
    def forward(self, x, S):
        x_S = torch.cat([x,S],1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        
        input_gate_H = torch.cat([x, S*R],1)
        H = self.gate_H(input_gate_H)
        
        output = ((1-G))*H + Z*S
        return output


class Net_DGM(nn.Module):

    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(Net_DGM, self).__init__()

        self.dim = dim_x
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))

        self.input_layer = nn.Sequential(nn.Linear(dim_x+1, dim_S, dtype=torch.double), self.activation)

        self.DGM1 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM2 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM3 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)

        self.output_layer = nn.Linear(dim_S, 1, dtype=torch.double)

    def forward(self,t,x):
        tx = torch.cat([t,x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx,S1)
        S3 = self.DGM2(tx,S2)
        S4 = self.DGM3(tx,S3)
        output = self.output_layer(S4)
        return output

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
            MSE_functional = loss_fn(pde, target_functional)# Should approach to zero
            
            # Terminal condtion
            x_terminal = x_batch
            t_terminal = torch.ones(batch_size, 1) * self.T
            u_terminal = self.net_dgm(t_terminal, x_terminal)
            terminal_condition = u_terminal - x_terminal.unsqueeze(2).transpose(1,2)@self.R@x_terminal.unsqueeze(2)
            target_terminal = torch.zeros(batch_size,1,dtype=float)
            MSE_terminal = loss_fn(terminal_condition, target_terminal)# Should approach to zero

            loss = MSE_terminal + MSE_functional
            loss.backward(retain_graph = True)
            optimizer.step()
            self.loss_list.append(loss)
            if i % 10 == 0:
                print(f"Epoch {i}: Loss = {loss.item()}")
                
    def get_loss(self):
            return self.loss_list
        
    def get_solution(self, t_batch, x_batch):
            return self.net_dgm(t_batch.unsqueeze(1),x_batch.squeeze(2))
        
    def error_aginst_MC(self,interval, t_batch, x_batch):
        lqr = LQR(self.H, self.M, self.sigma, self.C, self.D, self.R, self.T, 2)
        MC_test = lqr.MC_solution_afix(1000,1000,t_batch,x_batch)
        #print(MC_test)
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
                
    def get_error(self):           
        return self.error_list
