
"""
Created on Mon Apr 10 23:35:03 2023

@author: Linling Li and Xiaoya Yang
"""
import torch 
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from Ex_1_methods import LQR
from Ex_2_methods import FFN



class R1():
    def __init__(self, net1, T, N):
        self.net1 = net1
        self.T = T
        self.N = N
        
      
    def __equation(self, size,alpha):
        
        H = np.array([[1.0, 0.0],[0.0, 1.0]])
        M = np.array([[1.0, 0.0],[0.0, 1.0]])
        D = np.array([[0.1, 0.0],[0.0, 0.1]])
        C = np.array([[0.1, 0.0],[0.0, 0.1]])
        
        sigma = torch.tensor([[0.05],[0.05]],dtype = float)
        H = torch.tensor(H,dtype=float)
        M = torch.tensor(M,dtype=float)
        D = torch.tensor(D,dtype=float)
        C = torch.tensor(C,dtype=float)
        

        x = torch.cat((torch.rand([size, 1]) * self.T, torch.rand([size, 2], dtype=float) * self.N), dim=1)
        x = Variable(x, requires_grad = True)
        x_= torch.cat((x[:,1].reshape(-1,1),x[:,2].reshape(-1,1)),dim=1).unsqueeze(2)
   
    
        d = torch.autograd.grad(self.net1(x), x, grad_outputs = torch.ones_like(self.net1(x)), create_graph=True)
        dt = d[0][:, 0].reshape(-1, 1)  # transform the vector into a column vector
        dx = torch.cat((d[0][:,1].reshape(-1,1),d[0][:,2].reshape(-1,1)),dim=1).unsqueeze(2)
        g_dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(dx), create_graph=True)
        dxx = torch.cat((g_dxx[0][:,1].reshape(-1,1),g_dxx[0][:,2].reshape(-1,1)),dim=1).unsqueeze(2)

        f_matrix = (dx.transpose(1,2)@H@x_) +(dx.transpose(1,2)@M@alpha)+ (x_.transpose(1,2) @ C @x_) + alpha.transpose(1,2)@D@alpha
        # f = torch.diagonal(f_matrix, dim1=0)
        
        product = sigma@ sigma.transpose(0,1) @  dxx
        trace = torch.diagonal(product, dim1=1, dim2=2)
        
        diff_error = (dt + 0.5*trace + f_matrix)**2

        return diff_error

    def __boundary(self, size):
        R = np.array([[1.0, 0.0],[1.0, 0.0]])
        R = torch.tensor(R,dtype=float)
        x = torch.cat((torch.rand(size, 1)*self.T, torch.rand([size, 2],dtype=float)*self.N), dim=1)
        x = Variable(x, requires_grad = True)
        x_= torch.cat((x[:,1].reshape(-1,1),x[:,2].reshape(-1,1)),dim=1).unsqueeze(2)

        
        x_end = torch.cat((torch.ones(size, 1)*self.T, torch.rand([size, 2],dtype=float) * self.N), dim=1)
        xRx = x_.transpose(1,2)@R@x_
        end_error = (self.net1(x_end)-xRx)**2
        
        return end_error

    def loss_func(self, size, alpha):
        diff_error = self.__equation(size,alpha)
        end_error = self.__boundary(size)
        
        return torch.mean(diff_error + end_error )
    
    def net1_update(self):
        return self.net1

class H2():
    def __init__(self, net2, T, N):
        self.net2 = net2
        self.T = T
        self.N = N
    
    def __equation(self, size, net1):
        H = np.array([[1.0, 0.0],[0.0, 1.0]])
        M = np.array([[1.0, 0.0],[0.0, 1.0]])
        D = np.array([[0.1, 0.0],[0.0, 0.1]])
        C = np.array([[0.1, 0.0],[0.0, 0.1]])
        
        H = torch.tensor(H,dtype=float)
        M = torch.tensor(M,dtype=float)
        D = torch.tensor(D,dtype=float)
        C = torch.tensor(C,dtype=float)
        
        x = torch.cat((torch.rand([size, 1]) * self.T, torch.rand([size, 2], dtype=float) * self.N), dim=1)
        x = Variable(x, requires_grad = True)
        x_= torch.cat((x[:,1].reshape(-1,1),x[:,2].reshape(-1,1)),dim=1).unsqueeze(2)
    
        d = torch.autograd.grad(net1(x), x, grad_outputs = torch.ones_like(net1(x)), create_graph=True)
        dx = torch.cat((d[0][:,1].reshape(-1,1),d[0][:,2].reshape(-1,1)),dim=1).unsqueeze(2)
        a = self.net2(x)
        a = torch.tensor(a,dtype=float).unsqueeze(2)
        
        f_matrix = (dx.transpose(1,2)@H@x_) + (dx.transpose(1,2)@M@a)+ (x_.transpose(1,2) @ C @x_) + a.transpose(1,2)@D@a
        # f = torch.diagonal(f_matrix, dim1=0)
        
        diff_error = f_matrix**2

        return diff_error

    def loss_func(self, size, net1):
        diff_error = self.__equation(size,net1)
        
        return torch.mean(diff_error)
    
    def alpha_(self,size):
        x = torch.cat((torch.rand([size, 1]) * self.T, torch.rand([size, 2], dtype=float) * self.N), dim=1)
        x = Variable(x, requires_grad = True)
        alpha = self.net2(x)
        alpha = torch.tensor(alpha,dtype=float).unsqueeze(2)
        return alpha
    
    def net2_update(self):
        return self.net2
    
    def getdata(self,x_test):
        return self.net2(x_test)



class Train():
    def __init__(self, net1, R1eq, net2, H2eq, BATCH_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.net1 = net1
        self.model1 = R1eq
        self.errors2 = []
        self.net2 = net2
        self.model2 = H2eq

    def train(self, Epoch, epoch, lr, x_test):
        
        optimizer1 = optim.Adam(self.net1.parameters(), lr)
        optimizer2 = optim.Adam(self.net2.parameters(), lr)
        
        alpha = np.array([[1,1] for i in range(BATCH_SIZE)])
        alpha = torch.tensor(alpha,dtype=float).unsqueeze(2)
        v_output=[]
        a_output=[]
        for e in range(Epoch):
            for i in range(epoch):
                optimizer1.zero_grad()
                loss1 = self.model1.loss_func(self.BATCH_SIZE,alpha)
                loss1.backward(retain_graph = True)
                optimizer1.step()
                
            for j in range(epoch):
                optimizer2.zero_grad()
                loss2 = self.model2.loss_func(self.BATCH_SIZE,self.model1.net1_update())
                loss2.backward()
                loss2 = loss2.requires_grad_()
                optimizer2.step()
                
            error = self.model2.loss_func(self.BATCH_SIZE,self.model1.net1_update())
            self.errors2.append(error.detach())
               
                
            alpha = self.model2.alpha_(self.BATCH_SIZE)
            
           
            if e % 100 == 99:
                
                print("Epoch {} - lr {} -  loss: {}".format(e, lr, loss2.item()))
                
            netv=self.model1.net1_update()

               
            v_output.append(netv(x_test))
            
            a_output.append(self.model2.getdata(x_test))
        
        return v_output,a_output



    def get_errors(self):
        return self.errors2

    def save_model(self):
        torch.save(self.net2, 'net_model.pkl')
    
        



net1 =FFN(sizes=([3]+[400,300]+[1]))
net2 =FFN(sizes=([3]+[400,300]+[2]))


T= 10
N= 5
BATCH_SIZE = 2**10

R1equation = R1(net1, T, N)
H2equation = H2(net2, T, N)

T= 10
N= 1

 
t_ = torch.rand(100,1,dtype=float) * T
x_ = torch.rand(100, 2,dtype=float) * N

xx_test = torch.cat((t_, x_), dim=1)
xx_test = Variable(xx_test, requires_grad = True)


train = Train(net1, R1equation, net2, H2equation, BATCH_SIZE)

v_test_data,a_test_data= train.train(Epoch = 1000, epoch = 20, lr = 0.0005, x_test = xx_test)


loss = train.get_errors()
#Plot the training loss
fig = plt.figure()
plt.plot(loss, '-b', label='Errors')
plt.title('Training Loss', fontsize=10)
plt.show()

n = 2 # Define the dimension
H = np.array([[1.0, 0.0],[0.0, 1.0]])
M = np.array([[1.0, 0.0],[0.0, 1.0]])
sigma = torch.tensor([[0.05],[0.05]],dtype = float)
D = np.array([[0.1, 0.0],[0.0, 0.1]])
C = np.array([[0.1, 0.0],[0.0, 0.1]])
R = np.array([[1.0, 0.0],[0.0, 1.0]])

t_test, x_test = t_.squeeze(1), x_.unsqueeze(2)

#Generate data 
lqr = LQR(H, M, sigma, C, D, R, T, n)
v_test = lqr.value_function(t_test, x_test)
a_test = lqr.control_function(t_test, x_test)

errors_V = []
for i in range(len(v_test_data)):
    num = np.mean(((v_test_data[i]-v_test).detach().numpy())**2)
    errors_V.append(np.sqrt(num))
    
errors_a = []    
for i in range(len(a_test_data)):
    errors_a1 = a_test[0] - a_test_data[i][0]
    errors_a1 = ((errors_a1.detach().numpy()))**2
    errors_a2 = a_test[1] - a_test_data[i][1]
    errors_a2 = ((errors_a2.detach().numpy()))**2    
    num = np.mean(np.sqrt(errors_a1+errors_a2))
    errors_a.append(np.sqrt(num))
    
fig = plt.figure()
plt.plot(errors_V, label='Errors')
plt.title('V error', fontsize=10)
plt.show()
fig = plt.figure()
plt.plot(np.log(errors_a), label='Errors')
plt.title('a error', fontsize=10)
plt.show()
