"""
Amazon Top 5_000 Products

Online retail products, along with their attributes
such as brand, price, and description, used as a
product recommender via *large-scale* node classification.

This file is a loader for variations of the dataset.

"""

from typing import Optional

import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask, to_torch_coo_tensor


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Split and Load
#-----------------------------------------------------------------------------------------------------------------------------------------------------

"""
Dataset Loading
~~~~~~~~~~~~~~~

    `adjacency`: (str) encoding of edge indices and/or edge weights
        + `base`: edge indices filled if there is a connecting citation
        + `symm-norm`: edge indices by taking the symmetric norm of `base`
    `split`: (str) splits to use for learning
        + `fixed`: use the internal data splitting function
    #TODO: `folds`: (int) generate *n* random folds in randomly split data

"""

def snap_amz_dataloaders(
    adjacency : str = 'symm-norm',
    split : str = 'fixed_05',
    folds : Optional[int] = None,
):

    assert(adjacency in ['base','symm-norm']), f'Adjacency not recognized: {adjacency}'
    assert(split in ['fixed_05', 'fixed_06', 'fixed_07', 'fixed_08', 'fixed_09']), f'Split not recognized: {split}'


    transform = []
    if adjacency=='symm-norm': transform.append(T.GCNNorm())

    root = '/root/workspace/data/amazon/amazon-all/'
    graph = {}
    with open(root + 'adj_list.txt', 'r') as f:
        cur_idx = 0
        for row in f:
            row = row.strip().split()
            adjs = []
            for j in range(1, len(row)):
                adjs.append(int(row[j]))
            graph[cur_idx] = adjs
            cur_idx += 1
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.tocoo()

    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    # convert coo to torch_sparse

    idx_train = list(np.loadtxt(root + 'train_idx-' + str(portions_dict[split]) + '.txt', dtype=int))

    # !JB : bad practice to see that the validation and test sets are the same
    # !JB : https://github.com/SwiftieH/IGNN/blob/main/nodeclassification/utils.py#L327
    idx_val = list(np.loadtxt(root + 'test_idx.txt', dtype=int))
    idx_test = list(np.loadtxt(root + 'test_idx.txt', dtype=int))
    labels = np.loadtxt(root + 'label.txt')
    with open(root + 'meta.txt', 'r') as f:
        num_nodes, num_class = [int(w) for w in f.readline().strip().split()]

    # !JB : I don't know why the features are identity 
    # <https://github.com/SwiftieH/IGNN/blob/main/nodeclassification/utils.py#L333>
    features = torch.sparse.spdiags(torch.ones(num_nodes), offsets=torch.zeros(1,dtype=torch.long), shape=(num_nodes,num_nodes)).coalesce()
    
    labels = torch.FloatTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    train_mask = index_to_mask(idx_train, size=num_nodes)
    val_mask = index_to_mask(idx_val, size=num_nodes)
    test_mask = index_to_mask(idx_test, size=num_nodes)

    data = Data(
        x = features,
        edge_index = edge_index,
        y = labels,
        num_classes = num_class,
        num_nodes = num_nodes,
        train_mask = train_mask,
        val_mask = val_mask,
        test_mask = test_mask,
    )

    transform = T.Compose(transform)
    data = transform(data)
    adj = to_torch_coo_tensor(edge_index=data.edge_index, edge_attr=data.edge_weight)
    data.adj = adj

    return data, train_mask, val_mask, test_mask


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
Splits
~~~~~~~

"""


"""
Conversions
~~~~~~~~~~~

"""

portions_dict = {
    'fixed_05': 0.05,
    'fixed_06': 0.06,
    'fixed_07': 0.07,
    'fixed_08': 0.08,
    'fixed_09': 0.09
}


"""
Test
~~~~~~~~~~~

"""

if __name__ == '__main__':
    dataset, tr_mask, val_mask, test_mask = snap_amz_dataloaders(split='fixed_05')
    print(dataset)
    print(sum(tr_mask),sum(tr_mask)/len(tr_mask))
    print(sum(val_mask),sum(val_mask)/len(val_mask))
    print(sum(test_mask),sum(test_mask)/len(test_mask))