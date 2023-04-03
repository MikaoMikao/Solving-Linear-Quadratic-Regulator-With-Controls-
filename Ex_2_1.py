# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:32:58 2023

@author: 97896
"""
import torch
import numpy as np
from Net_DGM import Net_DGM
from FFN import FFN
from Ex_1_1_methods import LQR
import matplotlib.pyplot as plt

n = 2 # Define the dimension
H = np.array([[1.0, 0.0],[0.0, 1.0]])
M = np.array([[1.0, 0.0],[0.0, 1.0]])
sigma = torch.tensor([[0.05],[0.05]],dtype = float)
D = np.array([[0.1, 0.0],[0.0, 0.1]])
C = np.array([[0.1, 0.0],[0.0, 0.1]])
R = np.array([[1.0, 0.0],[0.0, 1.0]])

#Define the domain
T = 1
N=100

#Uniformly distributed samples
linspace1 = np.random.uniform(-3,3,100)
linspace2 = np.random.uniform(3,3,100)
linspace = [linspace1,linspace2]
x_train = torch.tensor(linspace, dtype = float).transpose(0,1).unsqueeze(2)
t_train = T*torch.rand(100,dtype = float)

#Generate training data
lqr = LQR(H, M, sigma, C, D, R, T, n)
v_train = lqr.value_function(t_train, x_train)
a_train = lqr.control_function(t_train, x_train)

#Ex 2.1

#Define the neural network
dgm = Net_DGM(2, 100)

#Define the optimizer as Adam optimizer
optimizer = torch.optim.Adam(dgm.parameters(), lr=0.01)

#Define the loss function
loss_fn = torch.nn.MSELoss()

best_loss = float('inf')
best_net = None
#Train the NN
n_epochs = 100
loss_list=[]
for epoch in range(n_epochs):
    optimizer.zero_grad()
    v_pred = dgm(t_train.unsqueeze(1),x_train.squeeze(2))
    loss = loss_fn(v_pred, v_train)
    loss.backward()
    optimizer.step()
    loss_list.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
    # Update best net
    if  loss < best_loss:
        best_loss = loss
        best_net = dgm

#plot the loss of training
#print(loss_list)
plt.plot(range(n_epochs), torch.tensor(loss_list))
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title('Loss of DGM Value Function')
plt.show()


#Ex 2.2
input_size = 2
output_size = 2
hidden_layers=[100,100]

ffn = FFN(sizes = [input_size+1]+hidden_layers+[output_size])

optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)

#Define the loss function
loss_fn = torch.nn.MSELoss()

best_loss = float('inf')
best_net = None
#Train the NN
n_epochs = 100
loss_list=[]
for epoch in range(n_epochs):
    optimizer.zero_grad()
    a_pred = ffn(torch.cat([t_train.unsqueeze(1),x_train.squeeze(2)],1))
    loss = loss_fn(a_pred, a_train)
    loss.backward()
    optimizer.step()
    loss_list.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
    # Update best net
    if  loss < best_loss:
        best_loss = loss
        best_net = ffn

#plot the loss of training
#print(loss_list)
plt.plot(range(n_epochs), torch.tensor(loss_list),'r')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title('Loss of FFN Markov Control')
plt.show()
