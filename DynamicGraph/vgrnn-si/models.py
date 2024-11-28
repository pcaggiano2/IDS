import torch
import utils as u
from argparse import Namespace
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
import math
import numpy as np


class Sp_GCN(torch.nn.Module):
    def __init__(self, args, activation):
        super().__init__()
        self.activation = activation
        self.num_layers = args.num_layers

        self.w_list = nn.ParameterList()
        for i in range(self.num_layers):
            if i == 0:
                w_i = Parameter(torch.Tensor(
                    args.feats_per_node, args.layer_1_feats))
                u.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(
                    args.layer_1_feats, args.layer_2_feats))
                u.reset_param(w_i)
            self.w_list.append(w_i)
        

    def forward(self, A_list, Nodes_list, nodes_mask_list):
        node_feats = Nodes_list[-1]
        # A_list: T, each element sparse tensor
        # take only last adj matrix in time
        Ahat = A_list[-1]
        # Ahat: NxN ~ 30k
        # sparse multiplication

        # Ahat NxN
        # self.node_embs = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        last_l = self.activation(Ahat.matmul(
            node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(
                Ahat.matmul(last_l.matmul(self.w_list[i])))
        return last_l


class Sp_Skip_GCN(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        self.W_feat = Parameter(torch.Tensor(
            args.feats_per_node, args.layer_1_feats))

    def forward(self, A_list, Nodes_list=None):
        node_feats = Nodes_list[-1]
        # A_list: T, each element sparse tensor
        # take only last adj matrix in time
        Ahat = A_list[-1]
        # Ahat: NxN ~ 30k
        # sparse multiplication

        # Ahat NxN
        # self.node_feats = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        l1 = self.activation(Ahat.matmul(node_feats.matmul(self.W1)))
        l2 = self.activation(Ahat.matmul(
            l1.matmul(self.W2)) + (node_feats.matmul(self.W3)))

        return l2


class Sp_Skip_NodeFeats_GCN(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)

    def forward(self, A_list, Nodes_list=None):
        node_feats = Nodes_list[-1]
        Ahat = A_list[-1]
        last_l = self.activation(Ahat.matmul(
            node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(
                Ahat.matmul(last_l.matmul(self.w_list[i])))
        # use node_feats.to_dense() if 2hot encoded input
        skip_last_l = torch.cat((last_l, node_feats), dim=1)
        return skip_last_l


class Sp_GCN_LSTM_A(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        self.rnn = nn.LSTM(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers
        )

        model_parameters = filter(
            lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"LSTM_A parameter {params}")

    def forward(self, A_list, Nodes_list=None, nodes_mask_list=None):
        last_l_seq = []
        for t, Ahat in enumerate(A_list):
            node_feats = Nodes_list[t].float()
            # assert (torch.count_nonzero(torch.isnan(node_feats))
            #         ) == 0, f"Node feat {t} containes nan"
            # A_list: T, each element sparse tensor
            # note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(
                node_feats.matmul(self.w_list[0])))
            # assert (torch.count_nonzero(torch.isnan(self.w_list[0]))
            #         ) == 0, f"w_list 0 containes nan"
            for i in range(1, self.num_layers):
                # assert (torch.count_nonzero(torch.isnan(self.w_list[i]))
                #         ) == 0, f"w_list {i} containes nan"
                # assert (torch.count_nonzero(torch.isnan(self.w_list[i]))
                #         ) == 0, f"last_l {i} containes nan"
                last_l = self.activation(
                    Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)
        # assert (torch.count_nonzero(torch.isnan(last_l_seq))
        #         ) == 0, f"input lstm contains nan"
        out, _ = self.rnn(last_l_seq, None)
        # assert (torch.count_nonzero(torch.isnan(out))
        #         ) == 0, f"out lstm contains nan"
        return out


class Sp_GCN_GRU_A(Sp_GCN_LSTM_A):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        print("GRU-A")
        self.rnn = nn.GRU(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers
        )
        model_parameters = filter(
            lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"GRU_A parameter {params}")


class Sp_GCN_LSTM_B(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        assert args.num_layers == 2, 'GCN-LSTM and GCN-GRU requires 2 conv layers.'
        self.rnn_l1 = nn.LSTM(
            input_size=args.layer_1_feats,
            hidden_size=args.lstm_l1_feats,
            num_layers=args.lstm_l1_layers
        )

        self.rnn_l2 = nn.LSTM(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers
        )
        self.W2 = Parameter(torch.Tensor(
            args.lstm_l1_feats, args.layer_2_feats))
        u.reset_param(self.W2)

    def forward(self, A_list, Nodes_list=None, nodes_mask_list=None):
        l1_seq = []
        l2_seq = []
        for t, Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            l1 = self.activation(Ahat.matmul(
                node_feats.matmul(self.w_list[0])))
            l1_seq.append(l1)

        l1_seq = torch.stack(l1_seq)

        out_l1, _ = self.rnn_l1(l1_seq, None)

        for i in range(len(A_list)):
            Ahat = A_list[i]
            out_t_l1 = out_l1[i]
            # A_list: T, each element sparse tensor
            l2 = self.activation(Ahat.matmul(out_t_l1).matmul(self.w_list[1]))
            l2_seq.append(l2)

        l2_seq = torch.stack(l2_seq)

        out, _ = self.rnn_l2(l2_seq, None)
        return out


class Sp_GCN_GRU_B(Sp_GCN_LSTM_B):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        self.rnn_l1 = nn.GRU(
            input_size=args.layer_1_feats,
            hidden_size=args.lstm_l1_feats,
            num_layers=args.lstm_l1_layers
        )

        self.rnn_l2 = nn.GRU(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers
        )


class Classifier(torch.nn.Module):
    def __init__(self, args, out_features=2, in_features=None):
        super(Classifier, self).__init__()
        activation = torch.nn.ReLU()

        if in_features is not None:
            num_feats = in_features
        elif args.experiment_type in ['sp_lstm_A_trainer', 'sp_lstm_B_trainer',
                                      'sp_weighted_lstm_A', 'sp_weighted_lstm_B']:
            num_feats = args.gcn_parameters['lstm_l2_feats'] * 2
        else:
            num_feats = args.gcn_parameters['layer_2_feats'] * 2
        print('CLS num_feats', num_feats)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=num_feats,
                                                       out_features=args.gcn_parameters['cls_feats']),
                                       activation,
                                       torch.nn.Linear(in_features=args.gcn_parameters['cls_feats'],
                                                       out_features=out_features))

    def forward(self, x):
        return self.mlp(x)


class Decoder(torch.nn.Module):

    class GraphConvolution(torch.nn.Module):
        """
        Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
        """

        def __init__(self, in_features, out_features, bias=True):
            super(Decoder.GraphConvolution, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Parameter(
                torch.FloatTensor(in_features, out_features))
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

        def reset_parameters(self):
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

        
        def forward(self, input, adj):
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output
    
        def __repr__(self):
            return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'

    class Attribute_Decoder(nn.Module):
        def __init__(self, nfeat, nhid, dropout):
            super(Decoder.Attribute_Decoder, self).__init__()

            self.gc1 = Decoder.GraphConvolution(nhid, nhid)
            self.gc2 = Decoder.GraphConvolution(nhid, nfeat)
            self.dropout = dropout

        def forward(self, x, adj, labels):
            assert (torch.count_nonzero(torch.isnan(x))
                    ) == 0, "Input Attribute decoder contains nan"
            x = F.relu(self.gc1(x, adj))
            assert (torch.count_nonzero(torch.isnan(x))
                    ) == 0, "GC1 decoder contains nan"
            x = F.dropout(x, self.dropout, training=False)  # self.training)
            # assert (torch.count_nonzero(torch.isnan(x))
            #         ) == 0, "Dropout decoder contains nan"
            x = self.gc2(x, adj)[labels == 1, :]
            # assert (torch.count_nonzero(torch.isnan(x))
            #         ) == 0, "GC2 decoder contains nan"
            return x

    class Structure_Decoder(nn.Module):
        def __init__(self, nhid, dropout):
            super(Decoder.Structure_Decoder, self).__init__()

            self.gc1 = Decoder.GraphConvolution(nhid, nhid)
            self.dropout = dropout

        def forward(self, x, adj, labels):
            x = self.gc1(x, adj)
            # x = F.dropout(x, self.dropout, training=self.training)
            # training=self.training)
            x = F.dropout(x, self.dropout, training=False)
            # x = x @ x.T
            # take element of interest
            x = x[labels == 1, :]
            # reconstruct only sub-network
            # x = torch.sigmoid(x @ x.T)
            x = x @ x.T

            return x

            # # assert (torch.count_nonzero(x < 0.0) ==
            #         0), "dot product less then zero"
            # x = torch.sigmoid(x)
            # print(self.gc1.weight)

    def __init__(self, args):
        super(Decoder, self).__init__()

        self.ad = Decoder.Attribute_Decoder(nfeat=args.decoder_parameters["ad_nfeat"],
                                            nhid=args.decoder_parameters["ad_nhid"],
                                            dropout=args.decoder_parameters["ad_dropout"]
                                            )#.to(f"cuda:{args.decoder_parameters['device_id']}")
        self.sd = Decoder.Structure_Decoder(nhid=args.decoder_parameters["sd_nhid"],
                                            dropout=args.decoder_parameters["sd_dropout"]
                                            )#.to(f"cuda:{args.decoder_parameters['device_id']}")
        model_parameters = filter(
            lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Decoder parameter {params}")

    def set_training(self, training):
        self.ad.training = training
        self.sd.training = training

    def forward(self, adj_mat, feature_attribute, labels, test=True):

        pred_attribute_mat = self.ad.forward(x=feature_attribute,
                                             adj=adj_mat,
                                             labels=labels)
        pred_adj_mat = self.sd.forward(x=feature_attribute,
                                       adj=adj_mat,
                                       labels=labels)

        return pred_attribute_mat, pred_adj_mat


class AnomalyDetector(torch.nn.Module):

    def __init__(self, gcn, head):
        super(AnomalyDetector, self).__init__()
        self.gcn = gcn
        self.head = head

        model_parameters = filter(
            lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Anomaly Detector parameters {params}")

    def set_training(self, training=True):
        self.training = training
        self.gcn.training = training
        self.head.set_training(training)

    def initialize_weights(self):
        self.gcn.initialize_weights()

    def forward_single_step(self, hist_adj_list, hist_ndFeats_list, mask_list):
        pass

    def parameters(self):
        return [*self.gcn.parameters(), *self.head.parameters()]

    def forward(self, hist_adj_list, hist_ndFeats_list, mask_list):
        nodes_embs = self.gcn(hist_adj_list,
                              hist_ndFeats_list,
                              mask_list)

        # run decoder inference
        pred_attribute_list = []
        pred_adj_list = []
        for t, nodes_e in enumerate(nodes_embs):
            # adj_mat, feature_attribute
            pred_attribute_mat, pred_adj_mat = self.head(adj_mat=hist_adj_list[t],
                                                         feature_attribute=nodes_e,
                                                         labels=mask_list[t])
            pred_attribute_list.append(pred_attribute_mat)
            pred_adj_list.append(pred_adj_mat)

        return pred_attribute_list, pred_adj_list
