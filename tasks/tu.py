#!/opt/conda/bin/python3
import os
import random
import sys
import time

import hydra
from omegaconf import OmegaConf
import wandb

import numpy as np
import torch
from torch.nn import Linear, Module, ReLU
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.graphgym import global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T

from datasets.tu_dataset import tu_dataloaders
sys.path.append('/root/workspace/MIGNN/agg/')
from _conv import MonotoneImplicitGraph, CayleyLinear, ReLU, TanH
from _deq import ForwardBackward, ForwardBackwardAnderson, PeacemanRachford, PeacemanRachfordAnderson, PowerMethod

"""
PyG TU Dataset
Splits: 80/10/10
"""

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Model
#----------------------------------------------------------------------------------------------------------------------------------------------------

class Model(Module):
    def __init__(self,
            in_channels,
            hidden_channels,
            out_channels,
            num_nodes,
            edge_index,
            edge_weight,
            dropout,
            tol,
            max_iter,
        )-> None:
        super(Model, self).__init__()
        
        self.act = ReLU()
        self.dropout = dropout
        self.enc = Linear(in_channels, hidden_channels)
        self.dec = Linear(hidden_channels, out_channels, bias=False)

        adj = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight[:,0], size=(num_nodes, num_nodes)).to('cuda')
        lin_module = CayleyLinear(hidden_channels, hidden_channels, num_nodes, invMethod='neumann-10', adj=adj, device='cuda')
        nonlin_module = ReLU()
        solver = PeacemanRachford(lin_module, nonlin_module, max_iter=max_iter, kappa=1.0, tol=tol, store=False)
        self.ig1 = MonotoneImplicitGraph(lin_module, nonlin_module, solver)
        pass

    def forward(self, x, edge_index, edge_weight, num_nodes, batch, *args, **kwargs):
        edge_weight = edge_weight.sum(dim=-1)
        adj = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=(num_nodes, num_nodes)).to('cuda')
        self.ig1.lin_module.set_adj(adj,None)
        x = self.enc(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ig1(x.T, *args, **kwargs).t()
        x = self.act(x)
        x = global_add_pool(x, batch)
        x = self.dec(x)
        return x

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper
#----------------------------------------------------------------------------------------------------------------------------------------------------

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Config/Model/Dataset
#----------------------------------------------------------------------------------------------------------------------------------------------------

def setup(cfg):
    args = cfg.setup

    seed_everything(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg['setup']['device'] = args['device'] if torch.cuda.is_available() else 'cpu'

    os.environ["WANDB_DIR"] = os.path.abspath(args['wandb_dir'])

    if args['sweep']:
        run_id = wandb.run.id
        cfg['load']['checkpoint_path']=cfg['load']['checkpoint_path'][:-3]+f'-ID({run_id}).pt'

    pass

#----------------------------------------------------------------------------------------------------------------------------------------------------

def load(cfg):
    args = cfg.load

    dataset, _, _, _, train_loader, val_loader, test_loader = tu_dataloaders(
        aug_dim = cfg.model['hidden_channels'],
        adjacency='sym-norm',
        batch_size=args['batch_size'],
    )

    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = dataset.num_features,
        hidden_channels = model_kwargs['hidden_channels'],
        out_channels = dataset.num_classes, 
        num_nodes = dataset[0].num_nodes,
        edge_index=dataset[0].edge_index,
        edge_weight=dataset[0].edge_weight,
        dropout = model_kwargs['dropout'],
        max_iter = model_kwargs['max_iter'],
        tol = model_kwargs['tol'],
    )

    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_loader, val_loader, test_loader

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, num_nodes=data.num_nodes, batch=data.batch)
    output = F.log_softmax(output, dim=-1)
    loss = F.nll_loss(output, data.y)
    loss.backward()
    optimizer.step()
    pred = output.max(dim=1)[1]
    acc = pred.eq(data.y).sum().item()
    return loss.item(), acc

@torch.no_grad()
def validate(cfg, data, model):
    model.eval()
    output = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, num_nodes=data.num_nodes, batch=data.batch)
    output = F.log_softmax(output, dim=-1)
    loss = F.nll_loss(output, data.y)
    pred = output.max(dim=1)[1]
    acc = pred.eq(data.y).sum().item()
    return loss.item(), acc

@torch.no_grad()
def test(cfg, data, model):
    model.eval()
    output = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, num_nodes=data.num_nodes, batch=data.batch)
    output = F.log_softmax(output, dim=-1)
    loss = F.nll_loss(output, data.y)
    pred = output.max(dim=1)[1]
    acc = pred.eq(data.y).sum().item()
    return loss.item(), acc

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, model, train_dl, val_dl):
    args = cfg.train

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'], amsgrad=False)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)

    best = 1e8
    bad_itr = 0
    for epoch in range(args['epochs']):

        model.train()
        train_loss, train_acc = 0, 0
        start = time.time()
        for data in train_dl:
            data = data.to(cfg.setup['device'])
            train_loss_, train_acc_ = train(cfg, data, model, optimizer)
            train_loss += train_loss_
            train_acc += train_acc_
        end = time.time()
        
        model.eval()
        val_loss, val_acc = 0, 0
        for i,data in enumerate(val_dl): 
            data = data.to(cfg.setup['device'])
            val_loss_, val_acc_ = validate(cfg, data, model)
            val_loss += val_loss_
            val_acc += val_acc_
        
        train_loss, val_loss = train_loss/len(train_dl.dataset), val_loss/len(val_dl.dataset)
        train_acc, val_acc = train_acc/len(train_dl.dataset), val_acc/len(val_dl.dataset)

        perf_metric = 1-val_acc #your performance metric here

        if perf_metric < best:
            best = perf_metric
            bad_itr = 0
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc,
                'best': best},
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr += 1

        wandb.log({
            'epoch':epoch,
            'train_loss':train_loss,
            'train_acc':train_acc,
            'val_loss':val_loss,
            'val_acc':val_acc,
            'best':best,
            'time':end-start,
        })
        print(f'Epoch({epoch:04d}) '
            f'| train({train_loss:.4f},{train_acc:.2f}) '
            f'| val({val_loss:.4f},{val_acc:.2f}) '
            f'| best({best:.4f}) '
            f'| time({end-start:.4f})'
        )

        if bad_itr>args['patience']:
            break

    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="/root/workspace/MIGNN/tasks/", config_name="tu")
def run_tu(cfg):
    """
    Execute run saving details to wandb server.
    """
    # Setup Weights and Bias
    wandb.init(
        dir='/root/workspace/out/',
        entity='pr-inspired-aggregation',
        mode='disabled',
        name='prgnn-'+cfg.load['dataset'],
        project='qm9',
        tags=['prgnn', cfg.load['dataset']],
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    model, train_dl, val_dl, test_dl = load(cfg)
    model = model.to(cfg.setup['device'])
    print(model)
    if cfg.setup['train']:
        run_training(cfg, model, train_dl, val_dl)

    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(cfg.setup['device'])
    for data in test_dl:
        data = data.to(cfg.setup['device'])
        test_loss, test_acc = test(cfg, data, model)

    print(f'test ({test_loss:.4f},{test_acc:.4f})')
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_tu()