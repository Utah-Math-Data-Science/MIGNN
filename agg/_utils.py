import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

def rms_norm(tensor,*args):
    return tensor.pow(2).mean().sqrt()

def rms_Wnorm(tensor,rtol,atol,y):
    return rms_norm(tensor/(atol+rtol*y.abs()))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def clip_gradient(model, clip_norm=10):
    """ clip gradients of each parameter by norm """
    for param in model.parameters():
        torch.nn.utils.clip_grad_norm_(param, clip_norm)
    return model

def tenseig(tens,device):
    Lambda_A, Q_A = torch.linalg.eig(tens)
    # if torch.eq(tens.t(),tens).all():
    #     Lambda_A, Q_A = torch.symeig(tens,eigenvectors=True)
    # else:
    #     Lambda_A, Q_A = torch.linalg.eig(tens)
    return Lambda_A.view(-1,1), Q_A

def speig(sp,device):
    sy = (abs(sp - sp.T) == 0).nnz
    if sy:
        Lambda_A, Q_A = scipy.linalg.eigh(sp.toarray())
    else:
        Lambda_A, Q_A = scipy.linalg.eig(sp.toarray())
    if device != 'cpu':
        Lambda_A = torch.from_numpy(Lambda_A).type(torch.FloatTensor).cuda()
        Q_A = torch.from_numpy(Q_A).type(torch.FloatTensor).cuda()
    else:
        Lambda_A = torch.from_numpy(Lambda_A).type(torch.FloatTensor)
        Q_A = torch.from_numpy(Q_A).type(torch.FloatTensor)
    return Lambda_A.view(-1,1), Q_A

def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G

def get_spectral_rad(sparse_tensor, tol=1e-5):
    """Compute spectral radius from a tensor"""
    A = sparse_tensor.data.coalesce().cpu()
    A_scipy = sp.coo_matrix((np.abs(A.values().numpy()), A.indices().numpy()), shape=A.shape)
    return np.abs(linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0]) + tol

def kronecker(A, B):
    return torch.einsum('ab,cd->acbd', A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def l_1_penalty(model, alpha=0.1):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += alpha * torch.sum(torch.abs(param))
    return regularization_loss

def projection_norm_inf(A, kappa=0.99, transpose=False):
    """ project onto ||A||_inf <= kappa return updated A"""
    # TODO: speed up if needed
    v = kappa
    if transpose:
        A_np = A.T.clone().detach().cpu().numpy()
    else:
        A_np = A.clone().detach().cpu().numpy()
    x = np.abs(A_np).sum(axis=-1)
    for idx in np.where(x > v)[0]:
        # read the vector
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a = np.sort(a_abs)

        s = np.sum(a) - v
        l = float(len(a))
        for i in range(len(a)):
            # proposal: alpha <= a[i]
            if s / l > a[i]:
                s -= a[i]
                l -= 1
            else:
                break
        alpha = s / l
        a = a_sign * np.maximum(a_abs - alpha, 0)
        # verify
        assert np.isclose(np.abs(a).sum(), v, atol=1e-4)
        # write back
        A_np[idx, :] = a
    A.data.copy_(torch.tensor(A_np.T if transpose else A_np, dtype=A.dtype, device=A.device))
    return  A

def sp_mtx_to_sp_tnsr(sp_mtrx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sp_mtrx = sp_mtrx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sp_mtrx.row, sp_mtrx.col)).astype(np.int64))
    values = torch.from_numpy(sp_mtrx.data)
    shape = torch.Size(sp_mtrx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_act(act_type):
    act_type = act_type.lower()
    if act_type == 'identity':
        return nn.Identity()
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'elu':
        return nn.ELU(inplace=True)
    elif act_type == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


@torch.enable_grad()
def regularize(z, x, reg_type, edge_index=None, norm_factor=None):
    z_reg = norm_factor*z

    if reg_type == 'Lap':  # Laplacian Regularization
        row, col = edge_index
        loss = scatter_add(((z_reg.index_select(0, row)-z_reg.index_select(0, col))**2).sum(-1), col, dim=0, dim_size=z.size(0))
        return loss.mean()
    
    elif reg_type == 'Dec':  # Feature Decorrelation
        zzt = torch.mm(z_reg.t(), z_reg)
        Dig = 1./torch.sqrt(1e-8+torch.diag(zzt, 0))
        z_new = torch.mm(z_reg, torch.diag(Dig))
        zzt = torch.mm(z_new.t(), z_new)
        zzt = zzt - torch.diag(torch.diag(zzt, 0))
        zzt = F.hardshrink(zzt, lambd = 0.5)
        square_loss = F.mse_loss(zzt, torch.zeros_like(zzt))
        return square_loss

    else:
        raise NotImplementedError

class Append_func(nn.Module):
    def __init__(self, coeff, reg_type):
        super().__init__()
        self.coeff = coeff
        self.reg_type = reg_type

    def forward(self, z, x, edge_index, norm_factor):
        if self.reg_type == '' or self.coeff == 0.:
            return z
        else:
            z = z if z.requires_grad else z.clone().detach().requires_grad_()
            reg_loss = regularize(z, x, self.reg_type, edge_index, norm_factor)
            grad = autograd.grad(reg_loss, z, create_graph=True)[0]
            z = z - self.coeff * grad
            return z

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def set_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)