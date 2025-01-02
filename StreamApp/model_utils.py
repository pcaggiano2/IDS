import pickle as pk
from torch.utils.data import Dataset
import scipy.sparse as sp
import os
import torch
import numpy as np
from functional import *
from torch_geometric.utils import to_dense_adj
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')


class Graph_dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path=dataset_path
        self.graph_list_file = os.listdir(dataset_path)

    def __len__(self):
        return len(self.graph_list_file)

    def __getitem__(self, index):
        with open (self.dataset_path+"/"+self.graph_list_file[index],"rb") as graph_file:
            graph = pk.load(graph_file)
            adj = graph['adj']
            feat = graph['node_features']
            truth = graph['node_labels'].flatten()
            loop_adj = adj + sp.eye(adj.shape[0])
            adj_norm = normalize_adj(loop_adj).toarray()
            return {"adj":torch.FloatTensor(adj_norm), "attrs":torch.FloatTensor(feat), "label":truth, "adj_labels":torch.FloatTensor(loop_adj)}

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def calculate_anomaly_score(model, attrs, adj, adj_label, alpha):
    A_hat, X_hat = model(attrs, adj)
    loss, _, _ = objective_function(attrs, A_hat, to_dense_adj(adj)[0], X_hat, alpha)
    return loss.detach().cpu().numpy()

def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def graph_batching(batch):
    return batch

def get_features_number(dataset_path):
    with open (dataset_path+"/"+os.listdir(dataset_path)[0],"rb") as graph_file:
        return pk.load(graph_file)['node_features'].shape[1]

def compute_losses(model, attrs, adj, adj_label, alpha):
    A_hat, X_hat = model(attrs, adj)
    loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, alpha)
     
    return torch.mean(loss).item(), struct_loss.item(), feat_loss.item()

def normalize_attrs(nodes_features, min_values, max_values):
    # print("PRE node features: ", nodes_features)
    nodes_features = (nodes_features - min_values) / (max_values - min_values)
    nodes_features = np.nan_to_num(nodes_features, nan=0.0)
    # print("POST node features: ", nodes_features)
    nodes_features = np.nan_to_num(nodes_features, nan=0.0, posinf=0.0)
    assert not np.any(np.isinf(nodes_features)), "node_embs contains inf"
    assert torch.sum(torch.isnan(torch.FloatTensor(nodes_features))
                     == True) == 0, "features contains nan"
    return nodes_features

from numpy import load

def find_min_max_range(dataset_path):
    with load(os.path.join(dataset_path, 'min.npz')) as file:
        min = file['arr_0']
    with load(os.path.join(dataset_path, 'max.npz')) as file:
        max = file['arr_0']

    return min, max