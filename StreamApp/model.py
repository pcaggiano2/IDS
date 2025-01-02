import torch
import torch.nn as nn
from torch_geometric.nn import GCN
from decoder import DotProductDecoder


class DOMINANT(nn.Module):
    """
    Deep Anomaly Detection on Attributed Networks

    DOMINANT is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The reconstruction mean squared
    error of the decoders are defined as structure anomaly score and
    attribute anomaly score, respectively.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim :  int
       Hidden dimension of model. Default: ``64``.
    encoder_layer : int, optional
       Total number of encoder layers. Default: ``1``.
    decoder_layer : int, optional
       Total number of decoder layers. Default: ``1``.
    dropout : float, optional
       Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
       Activation function if not None.
       Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to apply sigmoid to the structure reconstruction.
        Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    **kwargs : optional
        Additional arguments for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 encoder_layers=2,
                 decoder_layers=2,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 **kwargs):
        super(DOMINANT, self).__init__()

        # encoder layer shoud contain al least 1 layer
        assert encoder_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        # decoder layer shoud contain al least 1 layer
        assert decoder_layers >= 2, \
            "Number of layers must be greater than or equal to 2."

        # encoder
        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)
        # Attribute Reconstruction Decoder
        self.attr_decoder = backbone(in_channels=hid_dim,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                     out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)
        # Structure Reconstruction Decoder
        self.struct_decoder = DotProductDecoder(in_dim=hid_dim,
                                                hid_dim=hid_dim,
                                                num_layers=decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)

        self.emb = None

    def forward(self, x, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """
        # encode feature matrix
        # assert torch.sum(torch.isnan(x) == True) == 0, "input contains nan"
        self.emb = self.shared_encoder(x, edge_index)
        # assert torch.sum(torch.isnan(self.emb) == True) == 0, "embedding contains nan"

        # reconstruct feature matrix
        x_ = self.attr_decoder(self.emb, edge_index)

        # decode adjacency matrix
        s_ = self.struct_decoder(self.emb, edge_index)

        return x_, s_
