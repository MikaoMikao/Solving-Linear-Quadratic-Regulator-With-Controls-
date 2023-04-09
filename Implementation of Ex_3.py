# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 17:38:11 2023

@author: 97896
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from Ex3_1 import PDE, Train
from FFN import FFN
from Ex_1_1_methods import LQR

# Use FFN with 2 hidden layers of size 200
net =FFN(sizes=([3]+[200,200]+[1]))

# Generate test batch
T = 1
N = 1
size = 100
x_ = N*torch.rand(N,2,1,dtype = float)
t_ = T*torch.rand(N,dtype = float)

# Train the model and plot the trainning loss
PDE_equation = PDE(net, T, N)
train = Train(net, PDE_equation, BATCH_SIZE = size)
train.train(epoch = 1000, lr = 0.01)

loss = train.get_errors()

fig = plt.figure()
plt.plot(loss, '-b', label='Errors')
plt.title('Training Loss', fontsize=10)
plt.show()

# Use the same matrices from Exercise 1.1
n = 2 # Define the dimension
H = np.array([[1.0, 0.0],[0.0, 1.0]])
M = np.array([[1.0, 0.0],[0.0, 1.0]])
sigma = torch.tensor([[0.05],[0.05]],dtype = float)
D = np.array([[0.1, 0.0],[0.0, 0.1]])
C = np.array([[0.1, 0.0],[0.0, 0.1]])
R = np.array([[1.0, 0.0],[0.0, 1.0]])

t_test, x_test = t_, x_

# Obtain the MC solution with fixed optimal control
lqr = LQR(H, M, sigma, C, D, R, T, n)
MC_test = lqr.MC_solution_afix(1000,100,t_test, x_test)
print(MC_test)

# Plot the error against the MC solution
error_list=[]
error_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
n_epochs = 500

for epoch in range(n_epochs):
    optimizer.zero_grad()
    v_pred = net(torch.cat([t_test.unsqueeze(1),x_test.squeeze(2)],1))
    error = error_fn(v_pred, MC_test)
    error.backward()
    optimizer.step()
    error_list.append(error)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Error = {error.item()}")
        
plt.plot(range(n_epochs), torch.tensor(error_list),'r')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title('MSE Against MC Solution')
plt.show()        