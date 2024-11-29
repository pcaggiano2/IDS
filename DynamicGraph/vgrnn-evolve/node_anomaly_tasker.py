import taskers_utils as tu
import torch
import utils as u
from collections import OrderedDict
import pickle
import numpy as np
import scipy.sparse as sp
import os


class Anomaly_Detection_Tasker():
    def __init__(self, args, dataset):
        self.data = dataset
        self.is_static = False
        self.feats_per_node = dataset.feats_per_node
        self.num_classes = dataset.num_classes
        self.num_hist_steps = args.num_hist_steps
        self.adj_mat_time_window = args.adj_mat_time_window
        self.prepare_node_feats = self.build_prepare_node_feats(args, dataset)

    def build_get_node_feats(self, args, dataset):
        pass

    def build_prepare_node_feats(self, args, dataset):
        if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
            def prepare_node_feats(node_feats):
                return u.sparse_prepare_tensor(node_feats,
                                               torch_size=[dataset.num_nodes,
                                                           self.feats_per_node])
        # elif args.use_1_hot_node_feats:

        else:
            def prepare_node_feats(node_feats):
                return node_feats[0]  # I'll have to check this up

        return prepare_node_feats

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def get_adj_matrix(self, graph, weight=False, adj_mat_name='full_adj'):
        if adj_mat_name == 'full_adj':
            adj_mat = graph[adj_mat_name]
            adj_mat = torch.LongTensor(adj_mat)
            if not weight:
                vals = torch.ones(adj_mat.size(0), dtype=torch.long)

            out = torch.sparse.FloatTensor(
                adj_mat.t(), vals).coalesce()
            idx = out._indices().t()

            return {'idx': idx, 'vals': vals}
        else:
            return graph[adj_mat_name]

    def open_graph(self, capture_name, graph_type, graph_name, split):
        if split == "train" or split == "val" or split == "test_mixed" or split == "test_malicious" or split == "test_benign" or "iot23" in split:
            graph_base_folder = self.data.graph_base_folder
        elif split == "IoT_traces":
            graph_base_folder = self.data.graph_base_iot_traces_folder
        else:
            graph_base_folder = self.data.graph_base_iot_id20_folder

        if graph_type != None:
            graph_path = os.path.join(
                graph_base_folder, capture_name, self.data.representation, graph_type, graph_name)
        else:
            graph_path = os.path.join(
                graph_base_folder, capture_name, self.data.representation, graph_name)

        with open(graph_path, 'rb') as f:
            # print(graph_path)
            # print(f"Considering graph {graph_path.split('/')[-1]}")
            graph = pickle.load(f)
        return graph

    def get_sample(self, idx, start_indx, end_indx, graph_list, capture_name, graph_type, split):

        hist_adj_list = []
        hist_adj_list_norm = []
        hist_adj_list_partial = []
        hist_ndFeats_list = []
        hist_mask_list = []
        hist_node_labels = []

        # check if there are at least self.adj_mat_time_window graphs
        if (end_indx - start_indx)+1 < self.adj_mat_time_window:
            time_window = end_indx - start_indx
            if end_indx == start_indx:
                time_window = 1
        else:
            time_window = self.adj_mat_time_window

        if (end_indx - idx)+1 < time_window:
            # if (idx - time_window) < start_indx:
            #     start = start_indx
            # else:
            #     start = idx - time_window
            time_window = (end_indx - idx)+1
            start = idx
        else:
            start = idx

        assert time_window != 0, "Time window cannot be zero"

        for i in range(time_window):
            graph_data = self.open_graph(
                capture_name=capture_name,
                graph_type=graph_type,
                graph_name=graph_list[start+i][1],
                split=split)

            # 1. Create adj matrix
            # all edgess included from the beginning
            cur_adj = self.get_adj_matrix(graph=graph_data,
                                          adj_mat_name='full_adj')

            partial_adj = self.get_adj_matrix(
                graph=graph_data,
                adj_mat_name='adj')

            # 2. Create node mask
            node_mask = tu.get_node_mask(cur_adj=cur_adj,
                                         num_nodes=graph_data['n_nodes'])

            # 3. Create node features
            node_feats = self.get_node_features(graph=graph_data,
                                                num_nodes=graph_data['n_nodes'],
                                                capture=capture_name,
                                                graph_type=graph_type,
                                                graph_name=graph_list[start+i][1])

            # 4. Create node labels
            node_labels = self.get_node_labels(
                graph_data['label_new_indx_label'], idx)

            # 5. Normalize matrix
            cur_adj_norm = tu.normalize_adj(
                adj=cur_adj,
                num_nodes=graph_data['n_nodes'])

            hist_adj_list.append(cur_adj)
            hist_adj_list_norm.append(cur_adj_norm)
            hist_adj_list_partial.append(partial_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)
            hist_node_labels.append(node_labels)

        return {'idx': idx,
                'hist_adj_list': hist_adj_list,
                'hist_adj_list_norm': hist_adj_list_norm,
                'hist_adj_list_partial': hist_adj_list_partial,
                'hist_ndFeats_list': hist_ndFeats_list,
                'label_sp': hist_node_labels,
                'node_mask_list': hist_mask_list,
                'n_nodes': graph_data['n_nodes']}

    def get_node_labels(self, labels, idx):
        labels = torch.LongTensor(labels)
        label_idx = labels[:, 0]
        label_vals = labels[:, 1]
        return {'idx': label_idx,
                'vals': label_vals}

    def get_node_features(self, graph, num_nodes, capture, graph_type, graph_name):
        features = graph['node_features']
        # normalize features
        if self.data.normalize:
            features = (features-self.data.min_vector) / \
                (self.data.max_vector-self.data.min_vector)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0)
            max_feature_index = np.argmax(features)
            # Convert the flattened index to row and column indices
            row_index, col_index = np.unravel_index(
                max_feature_index, features.shape)

        new_features = np.zeros((num_nodes, self.feats_per_node))
        old_to_new_id_map = np.array(graph['features_new_old_map'])
        new_features[old_to_new_id_map[:, 1]
                     ] = np.array(features[old_to_new_id_map[:, 0]][:, :self.feats_per_node], dtype=np.float32)
        assert not np.any(np.isinf(new_features)), "new_features contains inf"

        return new_features

class Anomaly_Detection_Tasker_IoT_traces():
    def __init__(self, args, dataset):
        self.data = dataset
        self.is_static = False
        self.feats_per_node = dataset.feats_per_node
        self.num_classes = dataset.num_classes
        self.num_hist_steps = args.num_hist_steps
        self.adj_mat_time_window = args.adj_mat_time_window
        self.prepare_node_feats = self.build_prepare_node_feats(args, dataset)

    def build_get_node_feats(self, args, dataset):
        pass

    def build_prepare_node_feats(self, args, dataset):
        if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
            def prepare_node_feats(node_feats):
                return u.sparse_prepare_tensor(node_feats,
                                               torch_size=[dataset.num_nodes,
                                                           self.feats_per_node])
        # elif args.use_1_hot_node_feats:

        else:
            def prepare_node_feats(node_feats):
                return node_feats[0]  # I'll have to check this up

        return prepare_node_feats

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def get_adj_matrix(self, graph, weight=False, adj_mat_name='full_adj'):
        if adj_mat_name == 'full_adj':
            adj_mat = graph[adj_mat_name]
            adj_mat = torch.LongTensor(adj_mat)
            if not weight:
                vals = torch.ones(adj_mat.size(0), dtype=torch.long)

            out = torch.sparse.FloatTensor(
                adj_mat.t(), vals).coalesce()
            idx = out._indices().t()

            return {'idx': idx, 'vals': vals}
        else:
            return graph[adj_mat_name]

    def open_graph(self, capture_name, graph_type, graph_name, split):
        if split == "train" or split == "val" or split == "test_mixed" or split == "test_malicious" or split == "test_benign" or "iot_traces" in split:
            graph_base_folder = self.data.graph_base_folder
        elif split == "IoT23":
            graph_base_folder = self.data.graph_base_iot23_folder
        else:
            graph_base_folder = self.data.graph_base_iot_id20_folder

        if graph_type != None:
            graph_path = os.path.join(
                graph_base_folder, capture_name, self.data.representation, graph_type, graph_name)
        else:
            graph_path = os.path.join(
                graph_base_folder, capture_name, self.data.representation, graph_name)

        with open(graph_path, 'rb') as f:
            # print(graph_path)
            # print(f"Considering graph {graph_path.split('/')[-1]}")
            graph = pickle.load(f)
        return graph

    def get_sample(self, idx, start_indx, end_indx, graph_list, capture_name, graph_type, split):

        hist_adj_list = []
        hist_adj_list_norm = []
        hist_adj_list_partial = []
        hist_ndFeats_list = []
        hist_mask_list = []
        hist_node_labels = []

        # check if there are at least self.adj_mat_time_window graphs
        if (end_indx - start_indx)+1 < self.adj_mat_time_window:
            time_window = end_indx - start_indx
            if end_indx == start_indx:
                time_window = 1
        else:
            time_window = self.adj_mat_time_window

        if (end_indx - idx)+1 < time_window:
            # if (idx - time_window) < start_indx:
            #     start = start_indx
            # else:
            #     start = idx - time_window
            time_window = (end_indx - idx)+1
            start = idx
        else:
            start = idx

        assert time_window != 0, "Time window cannot be zero"

        for i in range(time_window):
            graph_data = self.open_graph(
                capture_name=capture_name,
                graph_type=graph_type,
                graph_name=graph_list[start+i][1],
                split=split)

            # 1. Create adj matrix
            # all edgess included from the beginning
            cur_adj = self.get_adj_matrix(graph=graph_data,
                                          adj_mat_name='full_adj')

            partial_adj = self.get_adj_matrix(
                graph=graph_data,
                adj_mat_name='adj')

            # 2. Create node mask
            node_mask = tu.get_node_mask(cur_adj=cur_adj,
                                         num_nodes=graph_data['n_nodes'])

            # 3. Create node features
            node_feats = self.get_node_features(graph=graph_data,
                                                num_nodes=graph_data['n_nodes'],
                                                capture=capture_name,
                                                graph_type=graph_type,
                                                graph_name=graph_list[start+i][1])

            # 4. Create node labels
            node_labels = self.get_node_labels(
                graph_data['label_new_indx_label'], idx)

            # 5. Normalize matrix
            cur_adj_norm = tu.normalize_adj(
                adj=cur_adj,
                num_nodes=graph_data['n_nodes'])

            hist_adj_list.append(cur_adj)
            hist_adj_list_norm.append(cur_adj_norm)
            hist_adj_list_partial.append(partial_adj)
            hist_ndFeats_list.append(node_feats)
            hist_mask_list.append(node_mask)
            hist_node_labels.append(node_labels)

        return {'idx': idx,
                'hist_adj_list': hist_adj_list,
                'hist_adj_list_norm': hist_adj_list_norm,
                'hist_adj_list_partial': hist_adj_list_partial,
                'hist_ndFeats_list': hist_ndFeats_list,
                'label_sp': hist_node_labels,
                'node_mask_list': hist_mask_list,
                'n_nodes': graph_data['n_nodes']}

    def get_node_labels(self, labels, idx):
        labels = torch.LongTensor(labels)
        label_idx = labels[:, 0]
        label_vals = labels[:, 1]
        return {'idx': label_idx,
                'vals': label_vals}

    def get_node_features(self, graph, num_nodes, capture, graph_type, graph_name):
        features = graph['node_features']
        # normalize features
        if self.data.normalize:
            features = (features-self.data.min_vector) / \
                (self.data.max_vector-self.data.min_vector)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0)
            max_feature_index = np.argmax(features)
            # Convert the flattened index to row and column indices
            row_index, col_index = np.unravel_index(
                max_feature_index, features.shape)

            # print(
            #     f"Max {np.max(features)}; {self.data.min_vector[col_index]}, {self.data.max_vector[col_index]}")
            # assert np.max(
            #     features) <= 1.0, f"max {np.max(features)}: {capture}-{graph_type}-{graph_name}"
            # assert np.count_nonzero(
            #     features < 0.0) == 0, f"input < 0.0: {np.max(features)}: {capture}-{graph_type}-{graph_name}"
            # assert np.count_nonzero(np.isnan(
            #     features)) == 0, f"Input nan: {np.max(features)}: {capture}-{graph_type}-{graph_name}"
        new_features = np.zeros((num_nodes, self.feats_per_node))
        old_to_new_id_map = np.array(graph['features_new_old_map'])
        new_features[old_to_new_id_map[:, 1]
                     ] = np.array(features[old_to_new_id_map[:, 0]][:, :self.feats_per_node], dtype=np.float32)
        assert not np.any(np.isinf(new_features)), "new_features contains inf"

        return new_features
if __name__ == '__main__':
    fraud_times = torch.tensor([10, 5, 3, 6, 7, -1, -1])
    idx = 6
    non_fraudulent = ((fraud_times > idx) + (fraud_times == -1)) > 0
    print(non_fraudulent)
    exit()
