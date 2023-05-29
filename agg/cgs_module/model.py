import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

sys.path.append('/root/workspace/MIGNN//agg/')
from _conv import MonotoneImplicitGraph, CayleyLinear, ReLU, TanH
from _deq import ForwardBackward, ForwardBackwardAnderson, PeacemanRachford, PeacemanRachfordAnderson, PowerMethod

from CGS.nn.MLP import MLP
from CGS.nn.MPNN import AttnMPNN


class porousMIGNN(nn.Module):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 lifted_dim: int,  # bias function input dim
                 hidden_dim: int,  # hidden state dim (state of the fp equation)
                 output_dim: int,
                 activation: str,
                 num_hidden_gn: int,
                 device,
                 mlp_num_neurons: list = [128],
                 reg_num_neurons: list = [64, 32], **kwargs):
        super(porousMIGNN, self).__init__()

        self.encoder = AttnMPNN(node_in_dim=node_dim,
                                edge_in_dim=edge_dim,
                                node_hidden_dim=64,
                                edge_hidden_dim=64,
                                node_out_dim=lifted_dim,
                                edge_out_dim=1,  # will be ignored
                                num_hidden_gn=num_hidden_gn,
                                node_aggregator='sum',
                                mlp_params={'num_neurons': mlp_num_neurons,
                                            'hidden_act': activation,
                                            'out_act': activation})

        lin_module = CayleyLinear(hidden_channels, hidden_channels, num_nodes, invMethod='neumann-10', adj=adj, device='cuda')
        nonlin_module = ReLU()
        solver = PeacemanRachford(lin_module, nonlin_module, max_iter=max_iter, kappa=1.0, tol=tol)
        self.fp_layer = MonotoneImplicitGraph(lin_module, nonlin_module, solver)
        self.decoder = MLP(hidden_dim, output_dim,
                           hidden_act=activation,
                           num_neurons=reg_num_neurons)

    def forward(self, g, nf, ef):
        """
        1. Transform input graph with node/edge features to the bias terms of the fixed point equations
        2. Solve fixed point eq
        3. Decode the solution with MLP.
        """

        unf, _ = self.encoder(g, nf, ef)

        adj = g.adj().to(nf.device)

        # adj = AugNorm(adj)
        adj =  LaplaceNorm(adj)
        self.fp_layer.lin_module.set_adj(adj,sp_adj=None)

        z,_ = self.fp_layer(unf.T)
        z = z.T
        pred = self.decoder(z)
        return pred