import torch 
import numpy as np
from torch.autograd import Variable

class PDE():
    def __init__(self, net, T, N):
        self.net = net
        self.T = T
        self.N = N

    def __equation(self, size):
        H = np.array([[1.0, 0.0],[0.0, 1.0]])
        M = np.array([[1.0, 0.0],[0.0, 1.0]])
        D = np.array([[0.1, 0.0],[0.0, 0.1]])
        C = np.array([[0.1, 0.0],[0.0, 0.1]])
        sigma = torch.tensor([[0.05],[0.05]],dtype=float)
        
        H = torch.tensor(H,dtype=float)
        M = torch.tensor(M,dtype=float)
        D = torch.tensor(D,dtype=float)
        C = torch.tensor(C,dtype=float)
        alpha = np.array([[1,1] for i in range(size)])
        alpha = torch.tensor(alpha,dtype=float).unsqueeze(2)

        x = torch.cat((torch.rand(size, 1)*self.T, torch.rand([size, 2],dtype=float)*self.N), dim=1)
        x = Variable(x, requires_grad = True)
        x_= torch.cat((x[:,1].reshape(-1,1),x[:,2].reshape(-1,1)),dim=1).unsqueeze(2)

    
        d = torch.autograd.grad(self.net(x), x, grad_outputs = torch.ones_like(self.net(x)), create_graph=True)
        dt = d[0][:, 0].reshape(-1, 1)  # transform the vector into a column vector
        dx = torch.cat((d[0][:,1].reshape(-1,1),d[0][:,2].reshape(-1,1)),dim=1).unsqueeze(2)
        g_dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(dx), create_graph=True)
        dxx = torch.cat((g_dxx[0][:,1].reshape(-1,1),g_dxx[0][:,2].reshape(-1,1)),dim=1).unsqueeze(2)
        
        f_matrix = (dx.transpose(1,2)@H@x_) + (dx.transpose(1,2)@M@alpha)+ (x_.transpose(1,2) @ C @x_) + alpha.transpose(1,2)@D@alpha
        
        product = sigma@ sigma.transpose(0,1) @  dxx
        trace = torch.diagonal(product, dim1=1, dim2=2)
        
        diff_error = torch.square(dt + 0.5*trace + f_matrix)

        return diff_error

    def __boundary(self, size):
        R = np.array([[1.0, 0.0],[1.0, 0.0]])
        R = torch.tensor(R,dtype=float)
        x = torch.cat((torch.rand(size, 1)*self.T, torch.rand([size, 2],dtype=float)*self.N), dim=1)
        x = Variable(x, requires_grad = True)
        x_= torch.cat((x[:,1].reshape(-1,1),x[:,2].reshape(-1,1)),dim=1).unsqueeze(2)

        
        x_end = torch.cat((torch.ones(size, 1)*self.T, torch.rand([size, 2],dtype=float) * self.N), dim=1)
        xRx = x_.transpose(1,2)@R@x_
        end_error = torch.square(self.net(x_end)-xRx)   
        
        return end_error

    def loss_func(self, size):
        diff_error = self.__equation(size)
        end_error = self.__boundary(size)
        
        return torch.mean(diff_error + end_error)
    
    def output(self, x):
        print(self.net(x))
        return self.net(x)
    

class Train():
    def __init__(self, net, PDE_equ, BATCH_SIZE):
        self.errors = []
        self.BATCH_SIZE = BATCH_SIZE
        self.net = net
        self.model = PDE_equ

    def train(self, epoch, lr):
        optimizer = torch.optim.Adam(self.net.parameters(), lr)
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.model.loss_func(self.BATCH_SIZE)
            loss.backward()
            optimizer.step()
            self.errors.append(loss.detach())
            if e % 100 == 0:
                print("Epoch {} - lr {} -  loss: {}".format(e, lr, loss.item()))
                
                
    def get_errors(self):
        return self.errors

    def save_model(self):
        torch.save(self.net, 'net_model.pkl')
    

