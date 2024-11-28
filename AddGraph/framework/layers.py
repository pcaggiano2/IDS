# _*_ coding:utf-8 _*_
# @author: Jiajie Lin
# @file: layers.py
# @time: 2020/03/08

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SpecialSpmmFunction(torch.autograd.Function):
    """
    Special function for sparse matrix multiplication during forward and backward propagation.
    Efficiently handles sparse matrices by using only sparse regions for backpropagation.
    """

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        # Create sparse matrix 'a' from indices and values
        assert not indices.requires_grad
        a = torch.sparse_coo_tensor(indices, values, shape)
        
        # Save tensors for backward pass
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)  # Perform sparse matrix multiplication

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        a, b = ctx.saved_tensors
        grad_values = grad_b = None

        # Compute gradient w.r.t. the values of sparse matrix 'a'
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]

        # Compute gradient w.r.t. the dense matrix 'b'
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)

        return None, grad_values, None, grad_b


class SpecialSpmm(Module):
    """
    A wrapper for the SpecialSpmmFunction to simplify its usage.
    """
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(Module):
    """
    Sparse Graph Attention Layer (GAT).
    This implements a GAT layer using sparse matrices for large-scale graphs.
    Reference: https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        
        self.in_features = in_features  # Input feature size (node embeddings)
        self.out_features = out_features  # Output feature size
        self.alpha = alpha  # Alpha parameter for LeakyReLU
        self.concat = concat  # Whether to apply non-linearity at the end (True if this is not the last layer)
        
        # Learnable weight matrix W (for node transformation)
        self.W = Parameter(torch.DoubleTensor(in_features, out_features))
        nn.init.xavier_normal_(self.W.data, gain=1.414)  # Xavier initialization for better gradient flow

        # Attention mechanism parameters (used to compute attention coefficients)
        self.a = Parameter(torch.DoubleTensor(1, 2 * out_features))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = dropout  # Dropout rate for regularization
        self.leakyrelu = nn.LeakyReLU(self.alpha)  # LeakyReLU activation for attention coefficients
        self.special_spmm = SpecialSpmm()  # Instance of SpecialSpmm for sparse operations

    def forward(self, input, adj):
        """
        Forward pass for the GAT layer.
        Args:
        - input (tensor): Node feature matrix (N x in_features).
        - adj (tensor): Sparse adjacency matrix of the graph.
        Returns:
        - h_prime (tensor): Output node features after attention (N x out_features).
        """
        device = 'cuda' if input.is_cuda else 'cpu'
        N = input.size(0)  # Number of nodes

        # Get edges from adjacency matrix (non-zero entries)
        edge = adj.nonzero().t()

        # Linear transformation of input features (Wx)
        h = torch.mm(input, self.W)
        h = F.dropout(h, self.dropout, training=self.training)  # Apply dropout

        # Compute attention coefficients for each edge
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # Concatenate node embeddings of each edge
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))  # Compute attention scores with LeakyReLU
        
        # Multiply attention scores by edge weights in the adjacency matrix
        adj_w = adj[edge[0], edge[1]]  # Extract edge weights
        edge_e = edge_e * adj_w  # Multiply edge weights with attention scores

        # Normalize the attention scores by row-wise sum
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=device))

        # Compute output features for each node (aggregate messages from neighbors)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        h_prime = h_prime.div(e_rowsum)  # Normalize by attention row-sum
        
        # Apply non-linearity (ELU) if this is not the last layer
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def loss(self):
        """
        L2 regularization loss for the weights of the layer.
        """
        return torch.norm(self.W, 2).pow(2) + torch.norm(self.a, 2).pow(2)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'
