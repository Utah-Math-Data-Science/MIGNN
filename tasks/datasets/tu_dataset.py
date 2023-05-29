"""
Technical University Dortmund Dataset (TUDataset)

A diverse set of graph data, ranging from small graphs with a few nodes
to large graphs with thousands of nodes, and from sparse to dense graphs.
The datasets cover various domains, such as bioinformatics, social networks,
chemistry, and computer vision. 

This file is a loader for variations of the dataset.

"""

from typing import Any, Optional

import torch
from torch.utils.data import random_split
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `name`: (str) label target for training
    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `default`: edge indices indicate default connection
        + `sym-norm`: symmetric normalization of default graph
    `batch_size`: (int) maximum batch size for graphs

"""
def tu_dataloaders(
    aug_dim: int,
    name : str = 'MUTAG',
    adjacency : str = 'default',
    batch_size : int = 128,
):

    assert(name in dominant_targets), f'Only dominant targets are currently supported.\nUnrecognized target: {name}'
    assert(adjacency in ['default','sym-norm']), f'Adjacency not recognized: {adjacency}'

    transform = [T.ToUndirected()]
    if adjacency=='sym-norm': transform.append(T.GCNNorm())

    dataset = TUDataset(
        root="/root/workspace/data/tu",
        name=name,
        transform=T.Compose(transform),
    )


    train_dataset, val_dataset, test_dataset = random_splits(dataset) #TODO: Implementation of splits

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


"""
Data Splits
~~~~~~~~~~~~~~~

    Split the dataset containing into training, validation and test sets:

    `random_splits`: #TODO: Working on this for graph classification
    
"""
def random_splits(dataset):

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)

    return random_split(dataset, [num_training, num_val, num_test])
    # return dataset[train_idxs], dataset[val_idxs], dataset[test_idxs]


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Transforms
#-----------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Look-Up Tables
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dictionaries
~~~~~~~~~~~~~

"""

dominant_targets = [
    'COX2',
    'MUTAG',
    'NCI1',
    'PTC_MR',
    'ENZYMES',
    'PROTEINS',
    'IMDB-BINARY',
    'REDDIT-BINARY',
    '',
    '',
]

targets = [
    'AIDS',
    'alchemy_full',
    #... There's so many
]