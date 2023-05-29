"""
Planetoid Citations

A collection of paper citation networks in the 
domains of Computer Science, Physics and Biology
used for node-classification tasks.

This file is a loader for variations of the dataset.

"""

from typing import Optional
from torch_geometric.data import Data

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce, to_dense_adj

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `name`: (str) name of dataset to use `cora`, `citeseer` or `pubmed`
    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `base`: edge indices filled if there is a connecting citation
        + `symm-norm`: edge indices by taking the symmetric norm of `base`
    `split`: (str) splits to use for learning
        + `public`: the public splits 
        + `geom-gcn`: the geom-gcn splits which contain 10 fold with 
        + `random-balanced-subsample`: randomly balanced with 20 training nodes per class 500 validation and 1000 test nodes.
        + `random-balanced-full`: randomly balanced with 20 training nodes per class 500 validation and the rest are test nodes.
        + `random-60/20/20`: randomly split with 60% training nodes 20% validation and 20% test nodes.
    #TODO: `folds`: (int) generate *n* random folds in randomly split data

"""

def planetoid_citation_dataloaders(
    name : str = 'cora',
    featurization : str = 'base',
    adjacency : str = 'base',
    split : str = 'public',
    folds : Optional[int] = None,
):

    name = name.lower()

    assert(name.lower() in ['cora','citeseer','pubmed']), f'Dataset not recognized: {name}'
    assert(featurization in ['base','norm']), f'Featurization not recognized: {featurization}'
    assert(adjacency in ['base','symm-norm', 'norm-lapl']), f'Adjacency not recognized: {adjacency}'
    assert(split in ['geom-gcn','public','random-balanced-subsample','random-balanced-full','random-60/20/20']), f'Split not recognized: {split}'

    split_vals = list(split.split('-'))
    split_vals.extend([None]*(3-len(split_vals)))
    split_name = split_vals[0]


    transform = []

    if adjacency=='symm-norm': transform.append(T.GCNNorm())
    if adjacency=='norm-lapl': transform.append(NormLaplacian())

    if featurization=='norm': transform.append(T.NormalizeFeatures())

    # Handles Random Splits
    # This block is really ugly and we should find a better way to handle data splits
    split_kwargs = {} 
    if 'balanced'==split_vals[1]:
        split_kwargs.update({
            'num_train_per_class':20,
            'num_val':500,
        })
        if 'subsample'==split_vals[2]:
            split_kwargs.update({'num_test':1000})
        elif 'full'==split_vals[2]:
            split_kwargs.update({'num_test':nodes[name.lower()]})
    elif 'gcn'==split_vals[1]:
        split_name = 'geom-gcn'
    else:
        split_kwargs.update({
            'num_train_per_class':int(0.6*nodes[name.lower()]/classes[name.lower()]),
            'num_val':int(0.2*nodes[name.lower()]),
            'num_test':int(0.2*nodes[name.lower()]),
        })

    dataset = Planetoid(root='/root/workspace/data/'+name,
        name = name,
        split = split_name,
        transform = T.Compose(transform),
        **split_kwargs
    )
    data = dataset[0]

    return dataset, data, data.train_mask, data.val_mask, data.test_mask 


"""
Transforms
~~~~~~~~~~

"""

class NormLaplacian(T.BaseTransform):
    r"""Applies the GCN normalization from the `"Semi-supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    paper (functional name: :obj:`gcn_norm`).

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.
    """
    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def __call__(self, data: Data) -> Data:
        # data = T.ToUndirected()(data)
        data = T.GCNNorm()(data)
        if data.edge_weight is None:
            data.edge_weight = -torch.ones(data.edge_index.shape[1])
        else:
            data.edge_weight = - data.edge_weight
        data = T.AddSelfLoops()(data)
        return data

"""
Conversions
~~~~~~~~~~~

"""

classes = {
    'cora':7,
    'citeseer':6,
    'pubmed':3,
}

nodes = {
    'cora':2708,
    'citeseer':3327,
    'pubmed':19717,
}


"""
Test
~~~~~~~~~~~

"""

if __name__ == '__main__':
    dataset, data, tr_mask, val_mask, test_mask = planetoid_citation_dataloaders(split='geom-gcn')
    print(dataset)
    print(data)
    print(sum(tr_mask),sum(tr_mask)/len(tr_mask))
    print(sum(val_mask),sum(val_mask)/len(val_mask))
    print(sum(test_mask),sum(test_mask)/len(test_mask))