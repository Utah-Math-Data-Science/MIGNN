#!/usr/bin/python3
from typing import Optional
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor

import os, sys, time

import hydra
from omegaconf import OmegaConf
import wandb

import numpy as np
import random
from sklearn import metrics
import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_dense_adj

from datasets.snap_amz import snap_amz_dataloaders
sys.path.append('/root/workspace/MIGNN/agg/')
from _conv import MonotoneImplicitGraph, CayleyLinear, ReLU, TanH
from _deq import ForwardBackward, ForwardBackwardAnderson, PeacemanRachford, PeacemanRachfordAnderson, PowerMethod

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

        adj = torch.sparse_coo_tensor(edge_index, edge_weight).to('cuda')
        lin_module = CayleyLinear(hidden_channels, hidden_channels, num_nodes, invMethod='neumann-1', adj=adj, device='cuda')
        nonlin_module = ReLU()
        solver = PeacemanRachfordAnderson(lin_module, nonlin_module, max_iter=max_iter, kappa=1.0, tol=tol)
        self.ig1 = MonotoneImplicitGraph(lin_module, nonlin_module, solver)

        pass

    def forward(self, x, edge_index, edge_weight, *args, **kwargs):
        x = self.enc(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ig1(x.t(), *args, **kwargs).t()
        x = self.act(x)
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

def clip_gradient(model, clip_norm=10):
    """ clip gradients of each parameter by norm """
    for param in model.parameters():
        torch.nn.utils.clip_grad_norm(param, clip_norm)
    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------

def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    '''
    binary_pred = preds
    binary_pred[binary_pred > 0.0] = 1
    binary_pred[binary_pred <= 0.0] = 0
    '''
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    print('total number of correct is: {}'.format(num_correct))
    #print('preds max is: {0} and min is: {1}'.format(preds.max(),preds.min()))
    #'''
    return metrics.f1_score(labels, binary_pred, average="macro"), metrics.f1_score(labels, binary_pred, average="micro")

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
    # Load data
    data, _, _, _ = snap_amz_dataloaders(
        adjacency='symm-norm',
        split = args['split'],
    )
    # Load model
    model_kwargs = OmegaConf.to_container(cfg.model)
    model = Model(
        in_channels = data.num_features,
        hidden_channels = model_kwargs['hidden_channels'],
        out_channels = data.num_classes, 
        num_nodes = data.num_nodes,
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        dropout = model_kwargs['dropout'],
        max_iter = model_kwargs['max_iter'],
        tol = model_kwargs['tol'],
    )
    # Load Model
    if os.path.exists(args['checkpoint_path']) and args['load_checkpoint']:
        checkpoint = torch.load(cfg.load['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, data

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Train/Validate/Test
#----------------------------------------------------------------------------------------------------------------------------------------------------

def train(cfg, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight)
    loss = F.binary_cross_entropy_with_logits(output[data.train_mask], data.y[data.train_mask])
    f1_max, f1_min = Evaluation(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # clip_gradient(model, clip_norm=0.5)
    optimizer.step()
    
    return loss.item(), f1_max, f1_min

def validate(cfg, data, model):
    model.eval()
    output = model(data.x, data.edge_index, data.edge_weight)
    loss = F.binary_cross_entropy_with_logits(output[data.val_mask], data.y[data.val_mask])
    
    f1_max, f1_min = Evaluation(output[data.val_mask], data.y[data.val_mask])
    return loss.item(), f1_max, f1_min


def test(cfg, data, model):
    model.eval()
    output = model(data.x, data.edge_index, data.edge_weight)
    loss = F.binary_cross_entropy_with_logits(output[data.test_mask], data.y[data.test_mask])
    
    f1_max, f1_min = Evaluation(output[data.test_mask], data.y[data.test_mask])
    return loss.item(), f1_max, f1_min
#----------------------------------------------------------------------------------------------------------------------------------------------------
# Main/Hydra/Fold/Train
#----------------------------------------------------------------------------------------------------------------------------------------------------

def run_training(cfg, data, model):
    args = cfg.train
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'], amsgrad=True)

    bad_itr = 0
    best = 1e8
    best_loss = 1e5

    for epoch in range(args['epochs']):
        start = time.time()
        train_loss, train_f1_max, train_f1_min = train(cfg, data, model, optimizer)
        end = time.time()
        val_loss, val_f1_max, val_f1_min = validate(cfg, data, model)

        perf_metric = train_loss # Can't use validation because of data leak

        if perf_metric<best:
            best = perf_metric
            best_loss = min(val_loss, best_loss)
            bad_itr=0
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                },
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr+=1


        # Log results
        wandb.log({'epoch':epoch,
            'train_loss':train_loss,
            'train_f1_max': train_f1_max,
            'train_f1_min': train_f1_min,
            'val_loss':val_loss,
            'val_f1_max': val_f1_max,
            'val_f1_min': val_f1_min,
            'perf_metric':perf_metric,
            'time':end-start,
            'best':best})
        print('Epoch: {:03d}, '
            'train_loss: {:.7f}, '
            'train_f1_max: {:2.2f}, '
            'train_f1_min: {:2.2f}, '
            'val_loss: {:.7f}, '
            'val_f1_max: {:2.2f}, '
            'val_f1_min: {:2.2f}, '
            'perf_metric: {:2.2f}, '
            'best: {:2.2f}, '
            'time: {:2.2f}, '
            ''.format(epoch+1, train_loss, 100*train_f1_max, 100*train_f1_min, val_loss, 100*val_f1_max, 100*val_f1_min, perf_metric, best, end-start))

        if bad_itr>args['patience']:
            break

        # Because val_data == test_data we cannot save based on the perf metric

    return best


#----------------------------------------------------------------------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="/root/workspace/MIGNN/tasks/", config_name="snap_amz.yaml")
def run_snap_amz(cfg):
    mode = 'online' if cfg.setup['sweep'] else 'disabled'
    wandb.init(entity='utah-math-data-science',
                project='pr-inspired-aggregation',
                mode=mode,
                name='prgnn-snap-amz',
                dir='/root/workspace/out/',
                tags=['snap-amz', 'prgnn'],
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    
    # Execute
    setup(cfg)
    print(OmegaConf.to_yaml(cfg))
    model, data= load(cfg)
    print(model)
    model.to(cfg.setup['device'])
    data.to(cfg.setup['device'])
    if cfg.setup['train']:
        run_training(cfg, data, model)

    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.setup['device'])
    test_loss, test_f1_max, test_f1_min = test(cfg, data, model)
    wandb.log({'test_loss':test_loss,
        'test_f1_max': test_f1_max,
        'test_f1_min': test_f1_min
    })
    print(f'Test: f1_max({100*test_f1_max:.2f}), f1_min({100*test_f1_min:.2f})')
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_snap_amz()