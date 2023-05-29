#!/usr/local/bin/python3
from time import perf_counter

import dgl
import hydra
import torch
import torch
import torch.optim as th_op
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

from CGS.experiments.porenet.generate_graph import generate_graphs_seq
from CGS.gnn.CGS.get_model import get_model
from CGS.gnn.IGNN.IGNN import IGNN
from model import porousMIGNN
from CGS.utils.test_utils import print_perf


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

warnings.filterwarnings('ignore')



@hydra.main(config_path="./config/", config_name='porenet')
def main(config=None):
    device = config.train.device
    # ORTHOGONAL
    model = porousMIGNN(node_dim=config.model.nf_dim,
                    edge_dim=config.model.ef_dim,
                    lifted_dim=config.model.num_heads,
                    hidden_dim=config.model.n_hidden_dim,
                    output_dim=config.model.sol_dim,
                    num_hidden_gn=config.model.num_hidden_gn,
                    activation=config.model.activation,
                    record=False,
                    device=device).to(device)

    opt = getattr(th_op, config.opt.name)(model.parameters(), lr=config.opt.lr)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=32)
    loss_fn = torch.nn.MSELoss()

    #TRAIN
    # 50 ~ 200 nodes (pores)
    ns_range = [13, 87] if config.train.tessellation == 'Delaunay' else [10, 40]

    model.train()
    for i in range(config.train.n_updates):
        if i % config.train.generate_g_every == 0:
            train_g = generate_graphs_seq(n_graphs=config.train.bs,
                                          nS_bd=ns_range,
                                          tessellation=config.train.tessellation)
            train_g = dgl.batch(train_g).to(device)

        start = perf_counter()
        train_nf, train_ef = train_g.ndata['feat'], train_g.edata['feat']
        train_y = train_g.ndata['target']
        train_pred = model(train_g, train_nf, train_ef)

        loss = loss_fn(train_pred, train_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        fit_time = perf_counter() - start

        # logging
        log_dict = {'iter': i,
                    'loss': loss.item(),
                    'fit_time': fit_time,
                    'forward_itr': model.fp_layer.frd_itr,
                    'lr': opt.param_groups[0]['lr']}
        print_perf(log_dict)

    # TEST
    ns_range = [83, 88] #200
    model.eval()
    for i in range(1000):
        test_g = generate_graphs_seq(n_graphs=1,
                                        nS_bd=ns_range,
                                        tessellation=config.train.tessellation)
        test_g = dgl.batch(test_g).to(device)

        start = perf_counter()
        test_nf, test_ef = test_g.ndata['feat'], test_g.edata['feat']
        test_y = test_g.ndata['target']
        test_pred = model(test_g, test_nf, test_ef)

        loss = loss_fn(test_pred, test_y)

        fit_time = perf_counter() - start

        # logging
        log_dict = {'test 200 ': loss.item()}
        print_perf(log_dict)

    ns_range = [139, 143] #300
    model.eval()
    for i in range(1000):
        test_g = generate_graphs_seq(n_graphs=1,
                                        nS_bd=ns_range,
                                        tessellation=config.train.tessellation)
        test_g = dgl.batch(test_g).to(device)

        start = perf_counter()
        test_nf, test_ef = test_g.ndata['feat'], test_g.edata['feat']
        test_y = test_g.ndata['target']
        test_pred = model(test_g, test_nf, test_ef)

        loss = loss_fn(test_pred, test_y)

        fit_time = perf_counter() - start

        # logging
        log_dict = {'test 300 ': loss.item()}
        print_perf(log_dict)


    ns_range = [195, 200] #400
    model.eval()
    for i in range(1000):
        test_g = generate_graphs_seq(n_graphs=1,
                                        nS_bd=ns_range,
                                        tessellation=config.train.tessellation)
        test_g = dgl.batch(test_g).to(device)

        start = perf_counter()
        test_nf, test_ef = test_g.ndata['feat'], test_g.edata['feat']
        test_y = test_g.ndata['target']
        test_pred = model(test_g, test_nf, test_ef)

        loss = loss_fn(test_pred, test_y)

        fit_time = perf_counter() - start

        # logging
        log_dict = {'test 400 ': loss.item()}
        print_perf(log_dict)

    ns_range = [260, 265] #500
    model.eval()
    for i in range(1000):
        test_g = generate_graphs_seq(n_graphs=1,
                                        nS_bd=ns_range,
                                        tessellation=config.train.tessellation)
        test_g = dgl.batch(test_g).to(device)

        start = perf_counter()
        test_nf, test_ef = test_g.ndata['feat'], test_g.edata['feat']
        test_y = test_g.ndata['target']
        test_pred = model(test_g, test_nf, test_ef)

        loss = loss_fn(test_pred, test_y)

        fit_time = perf_counter() - start

        # logging
        log_dict = {'test 500 ': loss.item()}
        print_perf(log_dict)

if __name__ == '__main__':
    main()