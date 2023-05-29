"""
Synthetic Chains Dataset

Single link chains of nodes with optionalities for multiple chains,
multiple classes and inter-class edges.

This file is a loader for variations of the dataset.

"""
from typing import Optional

import numpy as np
import torch

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, index_to_mask
import torch_geometric.transforms as T

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `name`: (str) name of dataset to use `computers` or `photo`
    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `base`: edge indices filled if there is a connecting citation
        + `symm-norm`: edge indices by taking the symmetric norm of `base`
    `split`: (str) splits to use for learning
        + `random-balanced-full`: randomly balanced with 20 training nodes per class 500 validation and the rest are test nodes.
    #TODO: `folds`: (int) generate *n* random folds in randomly split data

"""

def synth_chains_data(
    num_classes: int = 2,
    num_chains: int = 20,
    chain_len: int = 10,
    feature_dim: int = 100,
    noise: float = 0.0,
    split: str = 'fixed_20/100/200',
):
    
    assert split in ['fixed_20/100/200', 'fixed_05/10/85']

    r = np.random.RandomState(42)

    # Adjacency
    edge_index = np.array([[c*num_chains*chain_len + n*chain_len + i,
        c*num_chains*chain_len + n*chain_len + i + 1] 
        for c in range(num_classes) for n in range(num_chains) for i in range(chain_len-1)]
    )

    # Features
    features = r.uniform(-noise, noise, size=(num_classes, num_chains, chain_len, feature_dim))
    features[:, :, 0, :num_classes] += np.eye(num_classes).reshape(num_classes, 1, num_classes) # generate even number of each class
    features = features.reshape(-1,feature_dim)
    features = torch.tensor(features,dtype=torch.float32)

    # Labels
    labels = np.eye(num_classes).reshape(num_classes, 1, 1, num_classes).repeat(num_chains, axis=1).repeat(chain_len, axis=2) # one-hot labels
    labels = labels.reshape(-1, num_classes)
    labels = torch.tensor(labels,dtype=torch.long)
    labels = torch.max(labels, dim=1)[1]
    # labels = torch.tensor(labels,dtype=torch.long).squeeze()
 
    # Networkx Graph
    G = nx.DiGraph()
    G.add_edges_from(edge_index)
    for i,node in enumerate(G.nodes):
        feat_dict = {str(j):features[i,j] for j in range(feature_dim)}
        G.add_node(i,**feat_dict)


    # Splitting
    num_nodes = num_classes*num_chains*chain_len
    if split == 'fixed_20/100/200':
        train_split = 20
        val_split = 100
        test_split = 200
    elif split == 'fixed_05/10/85':
        train_split = int(num_nodes*0.05)
        val_split = int(num_nodes*0.1)
        test_split = num_nodes - train_split - val_split
    else:
        raise NotImplementedError

    idx_random = np.arange(num_nodes)
    r.shuffle(idx_random)
    train_mask = torch.tensor(idx_random[:train_split],dtype=torch.long)
    val_mask = torch.tensor(idx_random[train_split:train_split+val_split],dtype=torch.long)
    test_mask = torch.tensor(idx_random[train_split+val_split:train_split+val_split+test_split],dtype=torch.long)

    # PyG Conversion
    data = from_networkx(G,group_node_attrs=[str(i) for i in range(feature_dim)])
    data.x = data.x.to(torch.float32)
    data.y = labels.to(torch.long)
    data.num_classes = num_classes
    data.num_features = feature_dim
    data.train_mask = index_to_mask(train_mask, num_nodes)
    data.val_mask = index_to_mask(val_mask, num_nodes)
    data.test_mask = index_to_mask(test_mask, num_nodes)


    # data = T.NormalizeFeatures()(data)
    data = T.ToUndirected()(data)
    data = T.GCNNorm()(data)

    return data



def synth_color_counting(
    num_classes: int = 2,
    num_chains: int = 20,
    chain_len: int = 10,
    feature_dim: int = 50,
    noise: float = 0.0,
    colored_l: int = 10,
    split: str = 'fixed_05/10/85',
):

    assert split in ['fixed_20/100/200', 'fixed_05/10/85']

    r = np.random.RandomState(42)

    # Adjacency
    edge_index = np.array([[c*num_chains*chain_len + n*chain_len + i,
        c*num_chains*chain_len + n*chain_len + i + 1] 
        for c in range(num_classes) for n in range(num_chains) for i in range(chain_len-1)]
    )
    edge_index = torch.tensor(edge_index,dtype=torch.float32)

    # Features
    features = r.uniform(-noise, noise, size=(num_classes, num_chains, chain_len, feature_dim))

    for i in range(num_chains):
        random_num_1 = np.random.randint(1, np.ceil(colored_l)/2)
        # print(random_num_1)
        perm_colored = np.random.permutation(colored_l)
        # print(perm_colored)
        idx_1 = perm_colored[:random_num_1]
        idx_0 = perm_colored[random_num_1:]
        features[0, i, idx_0, :2] += np.array([1,0])
        features[0, i, idx_1, :2] += np.array([0,1])
        # print(f'chains with label 0: \nidx_0:{idx_0}, idx_1:{idx_1}')
    # print(features[0, :, idx_0, :])
    # print(idx_1, idx_0)

    # number of 0 nodes < number of 1 nodes  [:colored_l]
    for i in range(num_chains):
        random_num_0 = np.random.randint(1, np.ceil(colored_l)/2)
        # print(random_num_0)
        perm_colored = np.random.permutation(colored_l)
        # print(perm_colored)
        idx_0 = perm_colored[:random_num_0]
        idx_1 = perm_colored[random_num_0:]
        # print(idx_1, idx_0)
        features[1, i, idx_0, :2] += np.array([1,0])
        features[1, i, idx_1, :2] += np.array([0,1])
        # print(f'chains with label 1: \nidx_0:{idx_0}, idx_1:{idx_1}')

    features[:, :, 0, :num_classes] += np.eye(num_classes).reshape(num_classes, 1, num_classes) # generate even number of each class
    features = features.reshape(-1,feature_dim)
    features = torch.tensor(features,dtype=torch.float32)

    labels = np.eye(num_classes).reshape(num_classes, 1, 1, num_classes).repeat(num_chains, axis=1).repeat(chain_len, axis=2) # one-hot labels
    labels = labels.reshape(-1, num_classes)
    labels = torch.tensor(labels,dtype=torch.float32)

    # Networkx Graph
    # Splitting
    num_nodes = num_classes*num_chains*chain_len
    if split == 'fixed_20/100/200':
        train_split = 20
        val_split = 100
        test_split = 200
    elif split == 'fixed_05/10/85':
        train_split = int(num_nodes*0.05)
        val_split = int(num_nodes*0.1)
        test_split = num_nodes - train_split - val_split
    else:
        raise NotImplementedError

    idx_random = np.arange(num_nodes)
    r.shuffle(idx_random)
    train_mask = torch.tensor(idx_random[:train_split],dtype=torch.long)
    val_mask = torch.tensor(idx_random[train_split:train_split+val_split],dtype=torch.long)
    test_mask = torch.tensor(idx_random[train_split+val_split:train_split+val_split+test_split],dtype=torch.long)

    # PyG Conversion
    data = Data(x=features,
        edge_index=edge_index.T,
        num_nodes=num_classes*num_chains*chain_len,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask)

    data.x = data.x.to(torch.float32)
    data.y = data.y.to(torch.long)
    data.edge_index = data.edge_index.to(torch.long)
    data.num_classes = num_classes
    data.num_features = feature_dim
    data.train_mask = index_to_mask(train_mask, num_nodes)
    data.val_mask = index_to_mask(val_mask, num_nodes)
    data.test_mask = index_to_mask(test_mask, num_nodes)


    data = T.NormalizeFeatures()(data)
    data = T.ToUndirected()(data)
    data = NormLaplacian()(data)

    return data


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

if __name__=='__main__':
    data = synth_color_counting()
    print(data.x.shape, data.y.shape, data.edge_index.shape)