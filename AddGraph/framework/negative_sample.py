# _*_ coding:utf-8 _*_
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn


def update_adj(U_adj, snapshot, nodes):
    """
    Updates the adjacency matrices based on the current graph snapshot.

    Args:
        U_adj (torch.Tensor): The global adjacency matrix representing the overall connections.
        snapshot (list of tuples): The current edges in the graph.
        nodes (int): The total number of nodes in the graph.

    Returns:
        tuple: Updated global adjacency matrix (U_adj), current adjacency matrix (adj), and 
               aggregated adjacency matrix (Adj) for the snapshot.
    """
    Adj = torch.zeros((nodes, nodes), dtype=torch.int16)
    adj = torch.zeros((nodes, nodes), dtype=torch.int16)

    # Iterate through the edges in the snapshot to update the adjacency matrices
    for edge in snapshot:
        # Increment the counts in U_adj and adj for both directions (i -> j and j -> i)
        U_adj[edge[0].item() - 1][edge[1].item() - 1] += 1  # Convert to 0-based index
        U_adj[edge[1].item() - 1][edge[0].item() - 1] += 1  # Convert to 0-based index
        adj[edge[0].item() - 1][edge[1].item() - 1] += 1    # Convert to 0-based index
        adj[edge[1].item() - 1][edge[0].item() - 1] += 1    # Convert to 0-based index
        Adj[edge[0].item() - 1][edge[1].item() - 1] += 1

    return U_adj, adj, Adj


class negative_sample(nn.Module):
    """
    A class for generating negative samples for training using negative sampling.

    This module is designed to sample edges that are not present in the graph
    to create a more balanced training dataset, which is critical for anomaly detection tasks.
    """
    def __init__(self):
        super(negative_sample, self).__init__()

    def forward(self, adj, U_adj, snapshot, H, f, arg):
        """
        Generate negative samples and calculate the associated loss.

        Args:
            adj (torch.Tensor): The current adjacency matrix of the graph.
            U_adj (torch.Tensor): The global adjacency matrix.
            snapshot (torch.Tensor): Current graph edges.
            H (torch.Tensor): Node embeddings.
            f (function): A function to calculate loss based on node embeddings.
            arg (bool): A flag that determines the data handling method.

        Returns:
            torch.Tensor: The calculated negative loss for each sample.
        """
        if arg:
            # Convert snapshot to a tuple of edges if 'arg' is True
            data = tuple(map(tuple, snapshot.data.cpu().numpy()))
            D = adj.sum(1)  # Degree of each node in the current graph
            D_ = U_adj.sum(1)  # Degree of each node in the global graph
            D = D.data.cpu().numpy()
            D_ = D_.data.cpu().numpy()
        else:
            # Convert snapshot to a tuple of edges if 'arg' is False
            data = tuple(map(tuple, np.array(snapshot)))
            D = adj.sum(1)  # Degree of each node in the current graph
            D_ = U_adj.sum(1)  # Degree of each node in the global graph
            D = np.array(D)
            D_ = np.array(D_)

        len_snapshot = snapshot.shape[0]  # Number of edges in the current snapshot

        # Find the first node with no connections in U_adj
        th = (np.argwhere(D_ == 0) + 1)[0].item()

        n_loss = torch.zeros(len_snapshot)  # Initialize the negative loss tensor
        index = 0
        print('----------Begin----------')

        # Iterate through each edge in the snapshot to generate negative samples
        for i, j in tqdm(data):
            di = D[i - 1]
            dj = D[j - 1]
            pi = float(di) / (di + dj)  # Probability for node i
            pj = float(dj) / (di + dj)  # Probability for node j

            # Randomly choose a node to sample based on the computed probabilities
            d = np.random.choice(a=[j, i], size=1, replace=False, p=[pi, pj])
            d = d.item()
            flag = 1 if d == j else 0  # Track which node was chosen

            # Find nodes with no connections in U_adj
            d_ = np.argwhere(U_adj[d - 1] == 0) + 1
            d_ = (np.squeeze(d_))
            id1 = d_ < th
            d_ = d_[id1]
            id2 = d_ != d
            d_ = d_[id2]
            dn = np.random.choice(a=d_, size=1, replace=False)  # Randomly sample a negative node
            dn = dn.item()

            # Calculate loss based on node embeddings
            if flag:
                loss = f(hi=H[i - 1], hj=H[j - 1]) - f(hi=H[dn - 1], hj=H[d - 1])
            else:
                loss = f(hi=H[i - 1], hj=H[j - 1]) - f(hi=H[d - 1], hj=H[dn - 1])

            # Keep sampling until the loss is non-positive
            while loss > 0:
                d = np.random.choice(a=[j, i], size=1, replace=False, p=[pi, pj])
                d = d.item()
                flag = 1 if d == j else 0

                d_ = np.argwhere(U_adj[d - 1] == 0) + 1
                d_ = (np.squeeze(d_))

                id1 = d_ < th
                d_ = d_[id1]
                id2 = d_ != d
                d_ = d_[id2]
                dn = np.random.choice(a=d_, size=1, replace=False)
                dn = dn.item()

                if flag:
                    loss = f(hi=H[i - 1], hj=H[j - 1]) - f(hi=H[dn - 1], hj=H[d - 1])
                else:
                    loss = f(hi=H[i - 1], hj=H[j - 1]) - f(hi=H[d - 1], hj=H[dn - 1])

            n_loss[index] = loss  # Store the computed loss
            index += 1
            
        print('----------End----------')

        return n_loss  # Return the tensor containing negative losses
