# _*_ coding:utf-8 _*_
# @author: Jiajie Lin
# @file: model.py
# @time: 2020/03/08

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter 
from framework.layers import SpGraphAttentionLayer  # Importing the custom sparse GAT layer

class SpGAT(nn.Module):
    """
    Sparse Graph Attention Network (GAT) model with multi-head attention.
    """
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        """
        Initialize the SpGAT model.
        Args:
        - nfeat: Number of input features (per node).
        - nhid: Number of hidden units (per attention head).
        - nout: Number of output features (per node).
        - dropout: Dropout rate to apply during training.
        - alpha: Negative slope for LeakyReLU activation in attention mechanism.
        - nheads: Number of attention heads.
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout

        # Create multiple attention layers (multi-head attention)
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # Output layer combines the attention heads
        self.out_att = SpGraphAttentionLayer(nheads * nhid, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, Adj, adj):
        """
        Forward pass of SpGAT.
        Args:
        - x: Node feature matrix.
        - Adj: Sparse adjacency matrix.
        - adj: Dense adjacency matrix.
        """
        # Apply dropout to the input features
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Handle nodes with no neighbors
        flag = []
        D = adj.sum(1)  # Degree of each node
        for index, d in enumerate(D):
            if d.item() == 0:
                flag.append(index)
        
        # Apply multi-head attention and concatenate the results
        x_ = torch.cat([att(x, Adj) for att in self.attentions], dim=1)
        x_ = F.dropout(x_, self.dropout, training=self.training)

        # Output attention
        h = self.out_att(x_, Adj)
        
        # Handle nodes with no neighbors
        h[flag] = x[flag]
        
        # Apply ELU activation
        h = F.elu(h)
        return h

    def loss(self):
        """
        Compute the L2 loss for regularization.
        """
        loss = self.out_att.loss()
        for att in self.attentions:
            loss += att.loss()
        return loss


class HCA(nn.Module):
    """
    Hierarchical Contextual Attention (HCA) for capturing short-term patterns.
    """
    def __init__(self, hidden, dropout):
        super(HCA, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        
        # Learnable parameters for attention mechanism
        self.Q = Parameter(torch.DoubleTensor(hidden, hidden), requires_grad=True)
        self.r = Parameter(torch.DoubleTensor(hidden), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of Q and r using Xavier initialization.
        """
        nn.init.xavier_normal_(self.Q.data, gain=1.414)
        r_ = self.r.unsqueeze(0)
        nn.init.xavier_uniform_(r_.data, gain=1.414)

    def forward(self, C):
        """
        Forward pass of HCA.
        Args:
        - C: The sequence of hidden states (temporal data).
        """
        C_ = C.permute(1, 0, 2)  # Permute dimensions
        C_t = C_.permute(0, 2, 1)
        
        # Attention computation using learned parameters
        e_ = torch.einsum('ih,nhw->niw', self.Q, C_t)
        e_ = F.dropout(e_, self.dropout, training=self.training)
        e = torch.einsum('h,nhw->nw', self.r, torch.tanh(e_))
        e = F.dropout(e, self.dropout, training=self.training)
        
        # Attention weights
        a = F.softmax(e, dim=1)
        
        # Weighted sum of hidden states
        short = torch.einsum('nw,nwh->nh', a, C_)
        return short

    def loss(self):
        """
        L2 regularization for the learnable parameters.
        """
        return torch.norm(self.Q, 2).pow(2) + torch.norm(self.r, 2).pow(2)


class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) for combining short-term and long-term information.
    """
    def __init__(self, hidden, dropout):
        super(GRU, self).__init__()
        self.dropout = dropout
        
        # Parameters for GRU gates
        self.Up = Parameter(torch.DoubleTensor(hidden, hidden))
        self.Wp = Parameter(torch.DoubleTensor(hidden, hidden))
        self.bp = Parameter(torch.DoubleTensor(hidden))

        self.Ur = Parameter(torch.DoubleTensor(hidden, hidden))
        self.Wr = Parameter(torch.DoubleTensor(hidden, hidden))
        self.br = Parameter(torch.DoubleTensor(hidden))

        self.Uc = Parameter(torch.DoubleTensor(hidden, hidden))
        self.Wc = Parameter(torch.DoubleTensor(hidden, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters using Xavier initialization.
        """
        nn.init.xavier_normal_(self.Up.data, gain=1.414)
        nn.init.xavier_normal_(self.Wp.data, gain=1.414)
        nn.init.xavier_normal_(self.Ur.data, gain=1.414)
        nn.init.xavier_normal_(self.Wr.data, gain=1.414)
        nn.init.xavier_normal_(self.Uc.data, gain=1.414)
        nn.init.xavier_normal_(self.Wc.data, gain=1.414)

    def forward(self, current, short):
        """
        Forward pass of GRU. Combines current and short-term hidden states.
        Args:
        - current: Current hidden state.
        - short: Short-term hidden state.
        """
        # Update gate
        P = torch.sigmoid(torch.matmul(current, self.Up) + torch.matmul(short, self.Wp) + self.bp)
        P = F.dropout(P, self.dropout, training=self.training)

        # Reset gate
        R = torch.sigmoid(torch.matmul(current, self.Ur) + torch.matmul(short, self.Wr) + self.br)
        R = F.dropout(R, self.dropout, training=self.training)

        # Candidate hidden state
        H_tilda = torch.tanh(torch.matmul(current, self.Uc) + R * torch.matmul(short, self.Wc))
        H_tilda = F.dropout(H_tilda, self.dropout, training=self.training)

        # Final hidden state
        H = (1 - P) * short + P * H_tilda
        return H

    def loss(self):
        """
        L2 regularization for GRU parameters.
        """
        loss1 = torch.norm(self.Up, 2).pow(2) + torch.norm(self.Wp, 2).pow(2) + torch.norm(self.bp, 2).pow(2)
        loss2 = torch.norm(self.Ur, 2).pow(2) + torch.norm(self.Wr, 2).pow(2) + torch.norm(self.br, 2).pow(2)
        loss3 = torch.norm(self.Uc, 2).pow(2) + torch.norm(self.Wc, 2).pow(2)
        return loss1 + loss2 + loss3


class Score(nn.Module):
    """
    Anomaly score computation for edges between nodes.
    """
    def __init__(self, beta, mui, hidden, dropout):
        super(Score, self).__init__()
        self.a = Parameter(torch.DoubleTensor(hidden), requires_grad=True)
        self.b = Parameter(torch.DoubleTensor(hidden), requires_grad=True)
        self.beta = beta
        self.mui = mui
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters using Xavier initialization.
        """
        a_ = self.a.unsqueeze(0)
        nn.init.xavier_uniform_(a_.data, gain=1.414)
        b_ = self.b.unsqueeze(0)
        nn.init.xavier_uniform_(b_.data, gain=1.414)

    def forward(self, hi, hj):
        """
        Compute the anomaly score for an edge between node i and node j.
        Args:
        - hi: Embedding of node i.
        - hj: Embedding of node j.
        """
        s = self.a * hi + self.b * hj
        s = F.dropout(s, self.dropout, training=self.training)
        s_ = torch.norm(s, 2).pow(2)
        
        # Apply sigmoid to compute the score
        x = self.beta * s_ - self.mui
        score = 1.0 / (1 + torch.exp(-x))
        return score

    def loss(self):
        """
        L2 regularization for the score parameters.
        """
        return torch.norm(self.a, 2).pow(2) + torch.norm(self.b, 2).pow(2)


if __name__ == "__main__":
    # Testing the model with random inputs
    net = SpGAT(nfeat=11, nhid=15, nout=11, dropout=0.2, alpha=0.2, nheads=8)
    net2 = Score(beta=0.2, mui=0.6, hidden=11, dropout=0.2)
    h = torch.rand(8, 11)  # Random node features
    adj = torch.tensor([[0,1,1,0,0,0,1,0], [0,0,0,0,0,0,0,0], [0,0,0,1,0,2,0,0], 
                        [0,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,0], [1,0,0,0,0,0,1,0], 
                        [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]])
    h_ = net(h, adj)
    loss = net.loss()
    loss2 = net2(h_[1], h_[2])
    print(loss2)
    loss2.backward()
