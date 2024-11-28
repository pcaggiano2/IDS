import torch
import os
import json
import pickle as pk
import scipy.sparse as sp
import numpy as np
from torch_geometric.data import Dataset


class Graph_dataset(Dataset):
    def __init__(self, dataset_path, json_path, representation):
        super().__init__()
        self.dataset_path = dataset_path
        self.representation = representation
        self.graph_list_file = get_list_files(json_path, representation)
        self.n_nodes = get_total_nodes(os.path.join(self.dataset_path, self.graph_list_file[0]))
    def len(self):
        return len(self.graph_list_file)

    def get(self, idx):
        #print("Graph ", self.graph_list_file[idx])
        #print("Dataset path ", self.dataset_path)

        with open(os.path.join(self.dataset_path, self.graph_list_file[idx]), "rb") as graph_file:
            # print("Graph ", self.graph_list_file[idx])
            graph = pk.load(graph_file)
            #adj_matrix_old = graph['adj']
            # loop_adj = adj_matrix_old + sp.eye(adj_matrix_old.shape[0])
            # adj_norm = normalize_adj(loop_adj).toarray()
            # node_features = torch.FloatTensor(graph['node_features'])
            # adj_matrix = torch.FloatTensor(adj_norm)
            # edge_index = torch.nonzero(
            #     adj_matrix, as_tuple=False).t().contiguous()
            # print("edge index:", edge_index)
            edges_list = graph['full_adj']
            # labels = torch.FloatTensor(graph['node_labels'].flatten())
            # data = Data(x=node_features, edge_index=edge_index, y=labels)
            edges_label_list = graph['full_adj_labels']
            return edges_list, edges_label_list 

def get_total_nodes(graph_path):
    with open(graph_path, "rb") as graph_file:
        graph = pk.load(graph_file)
        return graph['n_nodes']

def create_path(capture, representation, set_type, graph_name):
    if set_type == "train" or set_type == "val" or set_type == "test_benign":
        path = os.path.join(capture, representation, "full_benign", graph_name)
    elif set_type == "test_mixed":
        path = os.path.join(capture, representation, "mixed", graph_name)
    elif set_type == "test_malicious":
        path = os.path.join(capture, representation,
                            "full_malicious", graph_name)
    return path


def get_list_files(json_path, representation):
    file_paths = []
    # read json object
    with open(json_path,  'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    # iter over json to create the original data path
    set_type = os.path.splitext(os.path.basename(json_path))[0]
    #for capture, file_lists in json_data.items():
    for capture, graph_list in json_data.items():

        #print(file_lists)
        #for graph_name_list in file_lists:
        #for graph_name in graph_name_list:
        for graph_name in graph_list:
            # capture/representation/full_benign or full_malicious or mixed/graph_x.pkl
            file_path = create_path(
                capture, representation, set_type, graph_name)
            file_paths.append(file_path)
    file_paths.sort()
    return file_paths

# def normalize_adj(adj):
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

