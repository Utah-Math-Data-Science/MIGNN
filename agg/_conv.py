from functools import reduce
import torch
import numpy as np
from torch.autograd.functional import jacobian
import torch.nn as nn

try:
    import geotorch
except:
    print('No geotorch - cayley not available')
import math
import random
import re
from torch_sparse import spmm
import torch
import torch.nn as nn

from _utils import tenseig, speig, get_G, get_spectral_rad, projection_norm_inf

class MonotoneImplicitGraph(nn.Module):
    def __init__(self, lin_module, nonlin_module, solver, **kwargs):
        super(MonotoneImplicitGraph, self).__init__()
        self.lin_module = lin_module
        self.nonlin_module = nonlin_module
        self.solver = solver
        pass
    def forward(self, x, *args, **kwargs):
        z = self.solver(x,**kwargs)
        return z

"""
MonotoneLinear - Wrapper class for linear methods used by fixed point solvers.
"""
class MonotoneLinear(nn.Module):

    def __init__(self, nfeat, nhid, num_node, device='cpu', adj=None, sp_adj=None, invMethod=None, kappa=0.9, **kwargs):
        super(MonotoneLinear,self).__init__()
        self.nfeat = nfeat
        self.invMethod = invMethod
        self.nhid = nhid
        self.num_node = num_node
        self.device = device
        self.kappa = kappa
        if adj is not None:
            self.set_adj(adj,sp_adj)
        pass

    def x_shape(self):
        return (self.nfeat, self.num_node)

    def z_shape(self):
        return (self.nhid, self.num_node)

    def forward(self, x, z):
        bias = self.bias(x)
        return bias + self.multiply(z)

    def set_adj(self,adj,sp_adj):
        self.A = adj
        self.num_node = adj.shape[0]
        self.At = torch.transpose(adj, 0, 1).coalesce()

        if self.invMethod == 'eig':
            assert (sp_adj is not None)
            self.Lambda_At, self.Q_At = speig(sp_adj.T,self.device)
        pass

    def init_inverse(self, alpha):
        if self.invMethod == 'eig':
            self.Lambda_W, self.Q_W = tenseig(self.W,self.device)
            self.G = get_G(self.Lambda_W, self.Lambda_At,alpha/(1+alpha))
            self.coef = 1/(1+alpha)

        elif self.invMethod == 'direct':
            Ik = torch.eye(self.nhid*self.num_node, dtype=self.A.dtype,
                        device=self.device)
            if self.At.is_sparse:
                self.inv = torch.inverse((1+alpha)*Ik-alpha*torch.kron(self.W,self.At.to_dense()))
            else:
                self.inv = torch.inverse((1+alpha)*Ik-alpha*torch.kron(self.W,self.At.contiguous()))

        elif re.match(r'^neumann-*',self.invMethod):
            self.coef = 1/(1+alpha)
            self.gamma = alpha/(1+alpha)
            pass

        else:
            raise NotImplementedError

    def inverse(self, z):
        if self.invMethod == 'eig':
            return self.coef * (self.Q_W @ (self.G * (self.Q_W.t() @ z @ self.Q_At)) @ self.Q_At.t())

        elif self.invMethod == 'direct':
            vec = self.inv @ z.flatten()
            return vec.reshape(self.nhid, self.num_node)

        elif re.match(r'^neumann-*',self.invMethod):
            return self.coef * self.neumannk(z,self.gamma,k=int(self.invMethod.split('-')[1]),transpose=False)

        else:
            raise NotImplementedError

    def inverse_transpose(self, g):
        if self.invMethod == 'eig':
            return self.coef * (self.Q_W @ (self.G * (self.Q_W.t() @ g @ self.Q_At)) @ self.Q_At.t())

        elif self.invMethod == 'direct':
            vec = self.inv.T @ g.flatten()
            return vec.reshape(self.nhid, self.num_node)

        elif re.match(r'^neumann-*',self.invMethod):
            return self.coef * self.neumannk(g,self.gamma,k=int(self.invMethod.split('-')[1]),transpose=True)

        else:
            raise NotImplementedError

    def neumannk(self,z,gamma,k,transpose):
        if k==0:
            return z
        else:
            znew = self.gamma*self.multiply_transpose(z) if transpose else self.gamma*self.multiply(z)
            return z + self.neumannk(znew,gamma,k=k-1,transpose=transpose)

    def init_W(self):
        self.W = self.get_W()
        pass

    def get_W(self):
        raise NotImplementedError

    def bias(self, x):
        raise NotImplementedError

    def multiply(self, z):
        raise NotImplementedError

    def multiply_transpose(self, g):
        raise NotImplementedError

"""
Cayley - Linear module of IGNN : W X A + B U A
W =  \sigma(\mu) * (I+C) @ (I-C)^{-1} @ D
"""
class CayleyLinear(MonotoneLinear):

    def __init__(self, nfeat, nhid, num_node, rho=.2, mu=None, **kwargs):
        super().__init__(nfeat,nhid,num_node, **kwargs)

        diag = [1]*int((1-rho)*nhid) + [-1]*int(rho*nhid)# [-1,1] values with %of -1s set by rho
        if len(diag)<nhid: diag += [1]
        random.shuffle(diag) #random shuffling
        self.D = torch.diag(torch.tensor(diag,device=self.device,dtype=torch.float))
        self.C = nn.Linear(nhid,nhid,bias=False,dtype=torch.float)
        geotorch.skew(self.C, 'weight')

        stdv = 1. / math.sqrt(nhid)
        self.Omega_1 = nn.Parameter(torch.FloatTensor(nhid, nfeat))
        self.Omega_1.data.uniform_(-stdv,stdv)
        self.I = torch.eye(self.D.shape[0], dtype=self.D.dtype,
                      device=self.D.device)

        if mu is None:
            self.mu = nn.Parameter(torch.ones(1,dtype=torch.float,device=self.device))
        else:
            self.mu = torch.tensor(mu,dtype=torch.float,device=self.device)
        pass

    def get_W(self):
        return self.kappa*torch.sigmoid(self.mu)*(self.I - self.C.weight) @ torch.inverse(self.I+self.C.weight)@self.D
        # return self.kappa*torch.sigmoid(self.mu)*(self.I - self.C.weight + self.C.weight.t()) @ torch.inverse(self.I+self.C.weight - self.C.weight.t())@self.D

    def bias(self, x):
        support_1 = torch.sparse.mm(x.t(), self.Omega_1.t()).t()
        support_1 = torch.sparse.mm(self.At, support_1.T).T
        B = support_1 
        return B

    def multiply(self, z):
        z_out = torch.spmm(z.t(),self.W.t()).t()
        z_out =  spmm(self.At.indices(), self.At.values(), self.At.size()[0], self.At.size()[1], z_out.T ).T
        return z_out

    def multiply_transpose(self, g):
        g_out = self.W.t() @ g
        g_out = torch.sparse.mm(self.A, g_out.T).T
        return g_out


"""
ReLU - Nonlinear function from monotone operator deq
"""
class ReLU(nn.Module):
    def forward(self, z):
        return torch.relu(z)
    def derivative(self, z):
        # return (z > torch.finfo(z.dtype).eps).type_as(z)
        return (z > 0).type_as(z)

class Ident(nn.Module):
    def forward(self, z):
        return z
    def derivative(self, z):
        return torch.ones(z.shape,dtype=z.dtype,device=z.device)

"TanH"
class TanH(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
    def forward(self, z):
        return torch.tanh(z)
    def derivative(self, z):
        return torch.ones_like(z)-torch.tanh(z)**2
    def inverse(self, z):
        return torch.arctanh(z)