from functools import reduce
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Function

from _utils import rms_Wnorm



class FixedPointSolver(nn.Module):
    def __init__(self, lin_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=50, update_alpha=False, alpha_factor=1, store=True, verbose=True, record=True, **kwargs):
        super().__init__()
        self.lin_module = lin_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.update_alpha = update_alpha
        self.alpha_factor = alpha_factor
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        # self.stats = SplittingMethodStats()
        # self.stats.set_options(alpha=alpha,tol=tol,max_iter=max_iter,verbose=True,**kwargs)
        self.record = record
        self.store = store
        # self.z0 = torch.zeros(self.lin_module.z_shape(), dtype=x.dtype, device=x.device)
        # self.u0 = torch.zeros(self.lin_module.z_shape(), dtype=x.dtype, device=x.device)
        self.z_init = None
        self.u_init = None
        pass

    def _fwd(self,z,u,alpha,bias):
        raise NotImplementedError

    def _bwd(self,z,u,alpha,bias):
        raise NotImplementedError

    def forward(self,x):
        self.lin_module.init_W()
        if not (isinstance(self,ForwardBackward) or isinstance(self,PowerMethod)):
            self.lin_module.init_inverse(self.alpha)

        with torch.no_grad():
            if self.update_alpha:
                if isinstance(self,SymmetricLinear) or isinstance(self,SkewLinear):
                    self.alpha = max((.9/self.alpha_factor)*torch.exp(self.lin_module.mu)/torch.norm(self.lin_module.W)**2,self.tol)
                else:
                    # self.alpha = max(.9/sum(*map(lambda x,y: x**y,torch.sigmoid(self.lin_module.mu),range(0,self.alpha_factor+1))),self.tol)
                    pass
                pass

            start = time.time()

            z = torch.zeros(self.lin_module.z_shape(), dtype=x.dtype, device=x.device) if self.z_init is None else self.z_init
            u = torch.zeros(self.lin_module.z_shape(), dtype=x.dtype, device=x.device) if self.u_init is None else self.u_init
            bias = self.lin_module.bias(x)

            err, conv = 1.0, 1.0
            it = 0

            it_alpha = self.alpha
            errs, resids = [], [1]
            # while (conv > self.tol and it < self.max_iter and not np.isnan(err)):
            while (err > self.tol and it < self.max_iter and not np.isnan(err)):
                zn,u=self._fwd(z,u,it_alpha,bias)
                fn = self.nonlin_module(self.lin_module(x,zn))
                resid = (zn-fn).norm(p=np.inf).item() 

                new_err = torch.norm(zn-z,np.inf).item()
                if new_err > err:
                    # it_alpha = min(1-it_alpha/1.2,1-1e-6)
                    it_alpha = max(it_alpha/10,1e-6)
                    # print('reduced',self.alpha,it_alpha)
                    # it_alpha = max(1/(2+it),10*self.tol)
                err = new_err

                errs.append(err)
                resids.append(resid)
                conv = abs(resids[-1]-resids[-2])
                z = zn
                it = it + 1

            if self.store:
                self.z_init = z.clone().detach()
                self.u_init = None if u is None else u.clone().detach()

        if self.record:
            Weigs = torch.linalg.eigvals(self.lin_module.W)
            Weigs = Weigs.abs()
            sorted_eigs = Weigs.sort()[0]
            sorted_eigs = [float(s.item()) for s in sorted_eigs]
            # self.stats.fwd_lWmax += [torch.max(Weigs)]
            # self.stats.fwd_lWmin += [torch.min(Weigs)]
            # self.stats.fwd_time.update(time.time() - start)
            # self.stats.fwd_iters.update(it)
            # self.stats.RESID.append(resids)
            # self.stats.ERR.append(errs)
            # print('Top 5 Eigs:',*sorted_eigs,sep=', ')
            # print(f'Forward: lam_max: {self.stats.fwd_lWmax[-1].item()}\t lam_min: {self.stats.fwd_lWmin[-1].item()}')
            # print("Forward:", it-1, err, 'Converged' if it<self.max_iter else 'Not Converged')

        zn = self.lin_module(x, z)
        zn = self.nonlin_module(zn)
        zn = self.Backward.apply(self, zn)
        return zn

    class Backward(Function):
        @staticmethod
        def forward(ctx, splitter, z):
            ctx.splitter = splitter
            ctx.save_for_backward(z)
            return z
        @staticmethod
        def backward(ctx, g):
            start = time.time()
            sp = ctx.splitter
            z_init = ctx.saved_tensors[0]
            j = sp.nonlin_module.derivative(z_init)
            I = torch.where(j==0)
            d = (1 - j) / j 
            v = j * g 
            z = torch.zeros(sp.lin_module.z_shape(), dtype=g.dtype, device=g.device)
            u = torch.zeros(sp.lin_module.z_shape(), dtype=g.dtype, device=g.device)

            err, conv = 1.0, 1.0
            errs,resids=[1],[1]
            it = 0
            it_alpha = sp.alpha
            while (err>sp.tol and it < sp.max_iter):

                zn,u = sp._bwd(z,u,it_alpha,d,v,j,I,g)
                new_err = torch.norm(zn-z,np.inf).item()
                fn = j*sp.lin_module.multiply_transpose(zn)+v
                resid = (zn-fn).norm(p=np.inf).item() / (1e-6+zn.norm().item())

                if new_err > err:
                    it_alpha = max(it_alpha/10,1e-3)
                    # it_alpha = min(1-it_alpha/1.2,1-1e-6)
                    # it_alpha = max(1/(2+it),10*sp.tol)
                err = new_err

                errs.append(err)
                resids.append(resid)
                z = zn
                it = it + 1

            if not isinstance(sp,PowerMethod):
                dg = sp.lin_module.multiply_transpose(zn)
                # dg = g + dg
                dg = v + dg
            else:
                dg = z

            # if sp.record:
                # sp.stats.bkwd_iters.update(it)
                # sp.stats.bkwd_time.update(time.time() - start)
                # Weigs = torch.linalg.eigvals(sp.lin_module.W.t())
                # Weigs = Weigs.abs()
            #     sp.stats.bwd_lWmax += [torch.max(Weigs)]
            #     sp.stats.bwd_lWmin += [torch.min(Weigs)]
            #     sp.stats.BRESID.append(resids)
            #     sp.stats.BERR.append(errs)
            #     sp.stats.dL += [g.norm()]
            #     sp.stats.dG += [dg.norm()]
            #     sp.stats.dW += [(dg @ torch.spmm(sp.lin_module.A, z_init.T)).norm()]
            #     print(f'Backward: lam_max: {sp.stats.bwd_lWmax[-1].item()}\t lam_min: {sp.stats.bwd_lWmin[-1].item()}')
            # print("Backward: ", it, err, 'Converged' if it<sp.max_iter else 'Not Converged')

            return None, dg

""" 
PowerMethod 
"""
class PowerMethod(FixedPointSolver):
    def __init__(self, lin_module, nonlin_module, **kwargs):
        super().__init__(lin_module,nonlin_module,**kwargs)
        pass
    def _fwd(self,z,u,alpha,bias):
        return self.nonlin_module(self.lin_module.multiply(z)+bias), None
    def _bwd(self,z,u,alpha,d,v,j,I,g):
        # zn=j*self.lin_module.multiply_transpose(z)+v
        zn=j*self.lin_module.multiply_transpose(z)+v
        return zn, None

"""
ForwardBackward - Modified from monotone operator deq
"""
class ForwardBackward(FixedPointSolver):
    def __init__(self, lin_module, nonlin_module, **kwargs):
        super().__init__(lin_module,nonlin_module,**kwargs)
        pass
    def _fwd(self,z,u,alpha,bias):
        zn = self.lin_module.multiply(z)
        zn = (1-alpha)*z + alpha*(zn+bias)
        zn = self.nonlin_module(zn)
        return zn, None
    def _bwd(self,z,u,alpha,d,v,j,I,g):
        zn = self.lin_module.multiply_transpose(z)
        zn = (1-alpha)*z + alpha*zn
        zn = (zn+alpha*(1+d)*v) / (1+alpha*d)
        # zn = (zn + alpha * v) / (1 + alpha * (1+ d)) #is correct?
        zn[I] = v[I]
        return zn,None

"""
PeacemanRachford - Splitting code from monotone operator deq
"""
class PeacemanRachford(FixedPointSolver):
    def __init__(self, lin_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=50, verbose=False, **kwargs):
        super().__init__(lin_module,nonlin_module,alpha,tol,max_iter,verbose,**kwargs)
        pass
    def _fwd(self,z,u,alpha,bias): # Don't use input alpha
        u_12 = 2*z - u
        z_12 = self.lin_module.inverse(u_12 + self.alpha * bias)
        up = 2 * z_12 - u_12
        zp = self.nonlin_module(up)
        return zp,up
    def _bwd(self,z,u,alpha,d,v,j,I,g): # Don't use input alpha
        u_12 = 2 * z - u
        z_12 = self.lin_module.inverse_transpose(u_12)
        u = 2 * z_12 - u_12
        zn = (u + self.alpha * (1 + d) * v) / (1 + self.alpha * d)
        zn[I] = v[I]
        # zn = (u + self.alpha * v) / (1 + sp.alpha * (1 + d)) # is correct?
        return zn, u


"""
DouglasRachford - modified for DR splitting
"""
class DouglasRachford(FixedPointSolver):

    def __init__(self, lin_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=50, verbose=False, **kwargs):
        super().__init__(lin_module,nonlin_module,alpha,tol,max_iter,verbose,**kwargs)
        pass
    def _fwd(self,z,u,alpha,bias):
        u_12 = 2 * z - u
        z_12 = self.lin_module.inverse(u_12 + self.alpha * bias)
        up = u + z_12 - z
        zp = self.nonlin_module(up)
        return zp,up
    def _bwd(self,z,u,alpha,d,v,j,I,g):
        u_12 = 2 * z - u
        z_12 = self.lin_module.inverse_transpose(u_12)
        u = u + z_12 - z
        zn = (u + self.alpha * (1 + d) * v) / (1 + self.alpha * d) 
        zn[I] = v[I]
        return zn,u


class AcceleratedSolver(FixedPointSolver):
    def __init__(self, lin_module, nonlin_module, m=5, beta=0.5, bkwd_factor=1, lam=1e-4, **kwargs):
        super().__init__(lin_module,nonlin_module,**kwargs)
        self.m = m # number of past iterates for anderson iteration
        self.beta = beta # anderson iteration weighting
        self.lam = lam # lambda for anderson iteration
        self.bkwd_factor = bkwd_factor
        pass

    def get_norms(self):
        return self.lin_module.get_norms()

    def forward(self,x):

        self.lin_module.init_W()
        if not (isinstance(self,ForwardBackwardAnderson) or isinstance(self,PowerMethodAnderson)):
            self.lin_module.init_inverse(self.alpha)

        with torch.no_grad():
            # print(.9/sum([torch.sigmoid(self.lin_module.mu)**k for k in range(0,self.alpha_factor+1)]))
            if self.update_alpha:
                if isinstance(self,SymmetricLinear) or isinstance(self,SkewLinear):
                    self.alpha = max((.9/self.alpha_factor)*torch.exp(self.lin_module.mu)/torch.norm(self.lin_module.W)**2,self.tol)
                else:
                    self.alpha = max(1/sum([torch.sigmoid(self.lin_module.mu)**k for k in range(0,self.alpha_factor)]),self.tol)
                    print(self.alpha)
                pass
            start = time.time()

            numel = reduce(lambda x,y : x*y, self.lin_module.z_shape(),1)
            Z = torch.zeros(self.m, numel, dtype=x.dtype, device=x.device) if self.z_init is None else self.z_init
            F = torch.zeros(self.m, numel, dtype=x.dtype, device=x.device) if self.u_init is None else self.u_init
            bias = self.lin_module.bias(x)

            errs, resids = [], []
            old_err = 1
            z0 = Z[0].view(self.lin_module.z_shape())
            f0, err, _, _ = self._fwd(z0, x, bias, errs, resids)
            F[0,:] = f0.reshape(numel)
            Z[1,:] = f0.reshape(numel)
            f1, err, _, _ = self._fwd(f0, x, bias, errs, resids)
            F[1,:] = f1.reshape(numel)

            H = torch.zeros(self.m + 1, self.m + 1, dtype=x[0].dtype, device=x[0].device)
            H[0, 1:] = H[1:, 0] = 1

            y = torch.zeros( self.m + 1, 1, dtype=x[0].dtype, device=x[0].device)
            y[0] = 1

            for it in range(2, self.max_iter):
                n = min(self.m, it)
                G = F[:n] - Z[:n]
                GGt = torch.mm(G, G.t())
                GGt /= torch.norm(GGt) + 1e-6 # normalizing doesn't affect the direction of the solution
                H[1:n+1, 1:n+1] = GGt + self.lam*torch.eye(n, dtype=x[0].dtype, device=x[0].device)
                alpha = torch.linalg.lstsq(
                    H[:n+1, :n+1],
                    y[:n+1] 
                )
                alpha = alpha.solution[1:n+1, 0]
                Z[it%self.m] = self.beta*(alpha @ F[:n]) + (1-self.beta)*(alpha@Z[:n])
                Zin = Z[it%self.m].view(self.lin_module.z_shape())
                Zo, err, resid, ret_z = self._fwd(Zin, x, bias, errs, resids)
                F[it%self.m] = Zo.reshape(numel)
                # if (resid < self.tol):
                # if ((resids[-1]-resids[-2])/resid < self.tol):
                if (err < self.tol):
                    break

        if self.store:
            self.z_init = Z
            self.F_init = F

        if self.record:
            Weigs = torch.linalg.eigvals(self.lin_module.W)
            Weigs = Weigs.abs()
            sorted_eigs = Weigs.sort()[0]
            sorted_eigs = [float(s.item()) for s in sorted_eigs]
            # self.stats.fwd_lWmax += [torch.max(Weigs)]
            # self.stats.fwd_lWmin += [torch.min(Weigs)]
            # self.stats.fwd_time.update(time.time() - start)
            # self.stats.fwd_iters.update(it)
            # self.stats.RESID.append(resids)
            # self.stats.ERR.append(errs)
            # print('Top 5 Eigs:',*sorted_eigs,sep=', ')
            # print(f'Forward: lam_max: {self.stats.fwd_lWmax[-1].item()}\t lam_min: {self.stats.fwd_lWmin[-1].item()}')
            # print("Forward:", it+1, err, 'Converged' if it+1<self.max_iter else 'Not Converged')

        zn = self.lin_module(x, ret_z)
        zn = self.nonlin_module(zn)
        zn = self.Backward.apply(self, zn)
        return zn
        
    class Backward(Function):
        @staticmethod
        def forward(ctx, splitter, z):
            ctx.splitter = splitter
            ctx.save_for_backward(z)
            return z
        @staticmethod
        def backward(ctx, g):
            start = time.time()
            sp = ctx.splitter
            z_init = ctx.saved_tensors[0]
            j = sp.nonlin_module.derivative(z_init)
            I = torch.where(j==0)
            d = (1 - j) / j 
            v = j * g 
            numel = reduce(lambda x,y : x*y, sp.lin_module.z_shape(),1)
            Z = torch.zeros(sp.m, numel, dtype=g.dtype, device=g.device)
            F = torch.zeros(sp.m, numel, dtype=g.dtype, device=g.device)

            errs, resids = [], [1]
            beta = sp.bkwd_factor * sp.beta
            z0 = Z[0].view(sp.lin_module.z_shape())
            f0, err, _ = sp._bwd(z0, g, d, v, I, errs)
            F[0,:] = f0.reshape(numel)
            Z[1,:] = f0.reshape(numel)
            f1, err, _ = sp._bwd(f0, g, d, v, I, errs)
            F[1,:] = f1.reshape(numel)

            H = torch.zeros(sp.m + 1, sp.m + 1, dtype=g[0].dtype, device=g[0].device)
            H[0, 1:] = H[1:, 0] = 1

            y = torch.zeros(sp.m + 1, 1, dtype=g[0].dtype, device=g[0].device)
            y[0] = 1

            for it in range(2, sp.max_iter):
                n = min(sp.m, it)
                G = F[:n] - Z[:n]
                GGt = torch.mm(G, G.t())
                GGt /= torch.norm(GGt) + 1e-6 # normalizing doesn't affect the direction of the solution
                H[1:n+1, 1:n+1] = GGt + sp.lam*torch.eye(n, dtype=g[0].dtype, device=g[0].device)
                alpha = torch.linalg.lstsq(
                    H[:n+1, :n+1],
                    y[:n+1] 
                )
                alpha = alpha.solution[1:n+1, 0]
                Z[it%sp.m] = beta*2*(alpha @ F[:n]) + (1-beta*2)*(alpha@Z[:n])
                Zin = Z[it%sp.m].view(sp.lin_module.z_shape())
                Zo, err, ret_z = sp._bwd(Zin, g, d, v, I, errs)
                fn = j*sp.lin_module.multiply_transpose(ret_z)+v
                resid = (ret_z-fn).norm(p=np.inf).item() 
                resids.append(resid)
                F[it%sp.m] = Zo.reshape(numel)
                # if (resid < sp.tol):
                # if ((resids[-1]-resids[-2])/resid < sp.tol):
                if (err < sp.tol):
                    break

            if not isinstance(sp,PowerMethodAnderson):
                dg = sp.lin_module.multiply_transpose(Zo)
                # dg = g + dg
                dg = v + dg
            else:
                dg = Zo

            # if sp.record:
            #     sp.stats.bkwd_iters.update(it)
            #     sp.stats.bkwd_time.update(time.time() - start)
            #     sp.stats.BRESID.append(resids)
            #     sp.stats.BERR.append(errs)
            #     Weigs = torch.linalg.eigvals(sp.lin_module.W.t())
            #     Weigs = Weigs.abs()
            #     sp.stats.bwd_lWmax += [torch.max(Weigs)]
            #     sp.stats.bwd_lWmin += [torch.min(Weigs)]
            #     sp.stats.BRESID.append(resids)
            #     sp.stats.BERR.append(errs)
            #     sp.stats.dL += [g.norm()]
            #     sp.stats.dG += [dg.norm()]
            #     sp.stats.dW += [(dg @ torch.spmm(sp.lin_module.A, z_init.T)).norm()]
            #     print(f'Backward: lam_max: {sp.stats.bwd_lWmax[-1].item()}\t lam_min: {sp.stats.bwd_lWmin[-1].item()}')
            # print("Backward: ", it+1, err, 'Converged' if it+1<sp.max_iter else 'Not Converged')

            return None, dg

class PowerMethodAnderson(AcceleratedSolver):
    def __init__(self, lin_module, nonlin_module, **kwargs):
        super().__init__(lin_module,nonlin_module,**kwargs)
    def _fwd(self, z, x, bias, errs, resids):
        zn =  self.nonlin_module(self.lin_module.multiply(z)+bias)
        # err = (zn - z).norm().item() / (zn.norm().item())
        # err = rms_Wnorm(zn-z,self.tol,self.tol,zn)
        err = torch.norm(zn-z,np.inf).item()
        fn = self.nonlin_module(self.lin_module.multiply(zn)+bias)
        resid = (zn - fn).norm().item() / (zn.norm().item())
        errs.append(err)
        resids.append(resid)
        return zn, err, resid, zn
    def _bwd(self, z, g, d, v, I, errs):
        zn=self.lin_module.multiply_transpose(z/(1+d))+v
        zn[I] = v[I]
        # err = (zn - z).norm().item() / (1e-6 + zn.norm().item())
        # err = rms_Wnorm(zn-z,self.tol,self.tol,zn)
        err = torch.norm(zn-z,np.inf).item()
        errs.append(err)
        return zn, err, zn

class ForwardBackwardAnderson(AcceleratedSolver):
    def __init__(self, lin_module, nonlin_module, **kwargs):
        super().__init__(lin_module,nonlin_module,**kwargs)
    def _fwd(self, z, x, bias, errs, resids):
        zn = self.lin_module.multiply(z)
        zn = (1 - self.alpha) * z + self.alpha * (zn + bias)
        zn = self.nonlin_module(zn)
        # err = (zn - z).norm().item() / (zn.norm().item())
        # err = rms_Wnorm(zn-z,self.tol,self.tol,zn)
        err = torch.norm(zn-z,np.inf).item()
        fn = self.nonlin_module(self.lin_module.multiply(zn)+bias)
        resid = (zn - fn).norm().item() / (zn.norm().item())
        errs.append(err)
        resids.append(resid)
        return zn, err, resid, zn
    def _bwd(self, z, g, d, v, I, errs):
        zn = self.lin_module.multiply_transpose(z)
        zn = (1-self.alpha)*z + self.alpha*zn
        zn = (zn+self.alpha*(1+d)*v) / (1+self.alpha*d)
        # zn = (zn + self.alpha * v) / (1 + self.alpha * (1+ d)) #is correct?
        zn[I] = v[I]
        # err = (zn - z).norm().item() / (1e-6 + zn.norm().item())
        # err = rms_Wnorm(zn-z,self.tol,self.tol,zn)
        err = torch.norm(zn-z,np.inf).item()
        errs.append(err)
        return zn, err, zn

"""
PeacemanRachford - Splitting code from monotone operator deq
"""
class PeacemanRachfordAnderson(AcceleratedSolver):
    def __init__(self, lin_module, nonlin_module, **kwargs):
        super().__init__(lin_module,nonlin_module,**kwargs)
        pass
    def _fwd(self, z, x, bias, errs, resids):
        u_12 = self.nonlin_module(z)
        z_12 = 2 * u_12 - z
        u = self.lin_module.inverse(z_12 + self.alpha * bias)
        zn = 2 * u - z_12
        ret_z = self.nonlin_module(zn)
        # err = (ret_z - u_12).norm().item() / (1e-6 + zn.norm().item())
        # err = rms_Wnorm(ret_z-u_12,self.tol,self.tol,ret_z)
        err = torch.norm(ret_z-u_12,np.inf).item()
        fn = self.nonlin_module(self.lin_module.multiply(ret_z)+bias)
        resid = (ret_z - fn).norm().item() / (zn.norm().item())
        errs.append(err)
        resids.append(resid)
        return zn, err, resid, ret_z
    def _bwd(self, z, g, d, v, I, errs):
        u_12 = (z + self.alpha * (1 + d) * v) / (1 + self.alpha * d) 
        u_12[I] = v[I]
        z_12 = 2 * u_12 - z
        u = self.lin_module.inverse_transpose(z_12)
        zn = 2 * u - z_12
        ret_z = (zn + self.alpha * (1 + d) * v) / (1 + self.alpha * d) 
        ret_z[I]=v[I]
        # err = (ret_z - u_12).norm().item() / (1e-6 + zn.norm().item())
        # err = rms_Wnorm(ret_z-u_12,self.tol,self.tol,ret_z)
        err = torch.norm(ret_z-u_12,np.inf).item()
        errs.append(err)
        return zn, err, ret_z

"""
DouglasRachfordAnderson - modified for DR splitting
"""
class DouglasRachfordAnderson(AcceleratedSolver):
    def __init__(self, lin_module, nonlin_module, **kwargs):
        super().__init__(lin_module,nonlin_module,**kwargs)
        pass
    def _fwd(self, z, x, bias, errs, resids):
        u_12 = self.nonlin_module(z)
        z_12 = 2 * u_12 - z
        u = self.lin_module.inverse(z_12 + self.alpha * bias)
        zn = z + u - u_12
        ret_z = self.nonlin_module(zn)
        # err = (ret_z - u_12).norm().item() / (1e-6 + zn.norm().item())
        # err = rms_Wnorm(ret_z-u_12,self.tol,self.tol,ret_z)
        err = torch.norm(ret_z-u_12,np.inf).item()
        fn = self.nonlin_module(self.lin_module.multiply(ret_z)+bias)
        resid = (ret_z - fn).norm().item() / (zn.norm().item())
        errs.append(err)
        resids.append(resid)
        return zn, err, resid, ret_z
    def _bwd(self, z, g, d, v, I, errs):
        u_12 = (z + self.alpha * (1 + d) * v) / (1 + self.alpha * d) 
        u_12[I] = v[I]
        z_12 = 2 * u_12 - z
        u = self.lin_module.inverse_transpose(z_12)
        zn = z + u - u_12
        ret_z = (zn + self.alpha * (1 + d) * v) / (1 + self.alpha * d) 
        ret_z[I]=v[I]
        # err = (ret_z - u_12).norm().item() / (1e-6 + zn.norm().item())
        # err = rms_Wnorm(ret_z-u_12,self.tol,self.tol,ret_z)
        err = torch.norm(ret_z-u_12,np.inf).item()
        errs.append(err)
        return zn, err, ret_z

"""
DouglasRachfordHalpern - modified for DR splitting
"""
class DouglasRachfordHalpern(AcceleratedSolver):
    def __init__(self, lin_module, nonlin_module, **kwargs):
        super().__init__(lin_module,nonlin_module,bkwd_factor=2,**kwargs)
        pass
    def _fwd(self, z, x, bias, errs, resids):
        u_12 = self.nonlin_module(z)
        z_12 = 2 * u_12 - z
        u = self.lin_module.inverse(z_12 + self.alpha * bias)
        zn = (1-1/(4+len(errs)))*z + u- u_12
        ret_z = self.nonlin_module(zn)
        # err = (ret_z - u_12).norm().item() / (1e-6 + zn.norm().item())
        err = rms_Wnorm(ret_z-u_12,self.tol,self.tol,ret_z)
        fn = self.nonlin_module(self.lin_module.multiply_transpose(ret_z))
        resid = (ret_z - fn).norm().item() / (zn.norm().item())
        errs.append(err)
        resids.append(resid)
        return zn, err, ret_z
    def _bwd(self, z, g, d, v, I, errs):
        u_12 = (z + self.alpha * (1 + d) * v) / (1 + self.alpha * d) 
        u_12[I] = v[I]
        z_12 = 2 * u_12 - z
        u = self.lin_module.inverse_transpose(z_12)
        zn = (1-1/(4+len(errs)))*z + u - u_12
        ret_z = (zn + self.alpha * (1 + d) * v) / (1 + self.alpha * d) 
        ret_z[I]=v[I]
        # err = (ret_z - u_12).norm().item() / (1e-6 + zn.norm().item())
        # err = rms_Wnorm(ret_z-u_12,self.tol,self.tol,ret_z)
        err = torch.norm(ret_z-u_12,np.inf).item()
        errs.append(err)
        return zn, err, ret_z