import argparse
import yaml
import torch
import numpy as np
import time
import random
import math
#Vgrnn
import pickle as pkl
import sys
import networkx as nx
import scipy.sparse as sp


def pad_with_last_col(matrix, cols):
    out = [matrix]
    pad = [matrix[:, [-1]]] * (cols - matrix.size(1))
    out.extend(pad)
    return torch.cat(out, dim=1)


def pad_with_last_val(vect, k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                     dtype=torch.long,
                     device=device) * vect[-1]
    vect = torch.cat([vect, pad])
    return vect


def sparse_prepare_tensor(tensor, torch_size, ignore_batch_dim=True):
    if ignore_batch_dim:
        tensor = sp_ignore_batch_dim(tensor)
    tensor = make_sparse_tensor(tensor,
                                tensor_type='float',
                                torch_size=torch_size)
    return tensor


def sp_ignore_batch_dim(tensor_dict):
    tensor_dict['idx'] = tensor_dict['idx'][0]
    tensor_dict['vals'] = tensor_dict['vals'][0]
    return tensor_dict


def aggregate_by_time(time_vector, time_win_aggr):
    time_vector = time_vector - time_vector.min()
    time_vector = time_vector // time_win_aggr
    return time_vector


def sort_by_time(data, time_col):
    _, sort = torch.sort(data[:, time_col])
    data = data[sort]
    return data


def print_sp_tensor(sp_tensor, size):
    print(torch.sparse.FloatTensor(sp_tensor['idx'].t(
    ), sp_tensor['vals'], torch.Size([size, size])).to_dense())


def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv, stdv)


def make_sparse_tensor(adj, tensor_type, torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                        adj['vals'].type(torch.float),
                                        tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                        adj['vals'].type(torch.float),
                                        tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                       adj['vals'].type(torch.long),
                                       tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')


def sp_to_dict(sp_tensor):
    return {'idx': sp_tensor._indices().t(),
            'vals': sp_tensor._values()}


class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''

    def __init__(self, adict):
        self.__dict__.update(adict)


def set_seeds(rank):
    seed = int(time.time())+rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower() == 'none':
        if type == 'int':
            return random.randrange(param_min, param_max+1)
        elif type == 'logscale':
            interval = np.logspace(np.log10(param_min),
                                   np.log10(param_max), num=100)
            return np.random.choice(interval, 1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param


def load_data(file):
    with open(file) as file:
        file = file.read().splitlines()
    data = torch.tensor([[float(r) for r in row.split(',')]
                        for row in file[1:]])
    return data


def load_data_from_tar(file, tar_archive, replace_unknow=False, starting_line=1, sep=',', type_fn=float, tensor_const=torch.DoubleTensor):
    f = tar_archive.extractfile(file)
    lines = f.read()
    lines = lines.decode('utf-8')
    if replace_unknow:
        lines = lines.replace('unknow', '-1')
        lines = lines.replace('-1n', '-1')

    lines = lines.splitlines()

    data = [[type_fn(r) for r in row.split(sep)]
            for row in lines[starting_line:]]
    data = tensor_const(data)
    # print (file,'data size', data.size())
    return data


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    # type=argparse.FileType(
    parser.add_argument(
        '--config_file', default='/user/apaolillo/dynamic_graphs/copys/experiments/150k_IoT23_etdg/parameters_egcn_h_anomaly_norm.yaml', type=str)
    #    mode='r'), help='optional, yaml file containing parameters to be used, overrides command #line parameters')
    #/user/apaolillo/dynamic_graphs/EvolveGCN/experiments/150k_IoT23_etdg/parameters_egcn_h_anomaly_norm.yaml
    parser.add_argument('--debug', action='store_true')
    # parser.add_argument()
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        print(args.config_file)
        # data = yaml.load(args.config_file)
        with open(args.config_file, 'r') as file:
            data = yaml.safe_load(file)
        delattr(args, 'config_file')
        # print(data)
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value

    args.learning_rate = random_param_value(
        args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
    # args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')
    args.num_hist_steps = random_param_value(
        args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')
    args.gcn_parameters['feats_per_node'] = random_param_value(
        args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
    args.gcn_parameters['layer_1_feats'] = random_param_value(
        args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
    # or args.gcn_parameters['layer_2_feats_same_as_l1'].lower() == 'true':
    if args.gcn_parameters['layer_2_feats_same_as_l1']:
        args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
    else:
        args.gcn_parameters['layer_2_feats'] = random_param_value(
            args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
    args.gcn_parameters['lstm_l1_feats'] = random_param_value(
        args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
    # or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower() == 'true':
    if args.gcn_parameters['lstm_l2_feats_same_as_l1']:
        args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
    else:
        args.gcn_parameters['lstm_l2_feats'] = random_param_value(
            args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
    args.gcn_parameters['cls_feats'] = random_param_value(
        args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')

    return args


#Vgrnn
from sklearn.metrics import roc_auc_score, average_precision_score


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        # with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
        #     u = pkl._Unpickler(rf)
        #     u.encoding = 'latin1'
        #     cur_data = u.load()
        #     objects.append(cur_data)

        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
        
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    # features = features / features.sum(-1, keepdim=True)
    # adding a dimension to features for future expansion
    if len(features.shape) == 2:
        features = features.view([1,features.shape[0], features.shape[1]])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false



def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, edges_pos, edges_neg, gdc):
    def GraphDC(x):
        if gdc == 'ip':
            return 1 / (1 + np.exp(-x))
        elif gdc == 'bp':
            return 1 - np.exp( - np.exp(x))

    J = emb.shape[0]

    # Predict on test set of edges
    edges_pos = np.array(edges_pos).transpose((1,0))
    emb_pos_sp = emb[:, edges_pos[0], :]
    emb_pos_ep = emb[:, edges_pos[1], :]

    # preds_pos is torch.Tensor with shape [J, #pos_edges]
    preds_pos = GraphDC(
        np.einsum('ijk,ijk->ij', emb_pos_sp, emb_pos_ep)
    )
    
    edges_neg = np.array(edges_neg).transpose((1,0))
    emb_neg_sp = emb[:, edges_neg[0], :]
    emb_neg_ep = emb[:, edges_neg[1], :]

    preds_neg = GraphDC(
        np.einsum('ijk,ijk->ij', emb_neg_sp, emb_neg_ep)
    )

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(preds_pos.shape[-1]), np.zeros(preds_neg.shape[-1])])
    
    roc_score = np.array(
        [roc_auc_score(labels_all, pred_all.flatten()) \
            for pred_all in np.vsplit(preds_all, J)] 
    ).mean()
    
    ap_score = np.array(
        [average_precision_score(labels_all, pred_all.flatten()) \
            for pred_all in np.vsplit(preds_all, J)]
    ).mean()

    return roc_score, ap_score
