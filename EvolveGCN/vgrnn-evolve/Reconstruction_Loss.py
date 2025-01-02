import torch
import utils as u
import torch.nn.functional as F


class ReconstructionLoss(torch.nn.Module):

    def __init__(self, args, dataset):
        super().__init__()
        self.weight = args.loss["weight"]
        self.pos_weight_a = args.loss["pos_weight_a"]
        self.pos_weight_s = args.loss["pos_weight_s"]
        self.bce_s = args.loss["bce_s"]

    def forward(self, pred_adj, gt_adj, pred_attri, gt_attri, node_mask, partial_mat, test=True):
        # select sub-graph
        # if test:
        #     score, attr_error, stru_error = self.objective_function(ground_truth_node_feat=gt_attri,
        #                                                             reconstructed_node_feat=pred_attri,
        #                                                             ground_truth_node_struct=gt_adj,
        #                                                             reconstructed_node_struct=pred_adj,
        #                                                             alpha=self.weight,
        #                                                             pos_weight_a=self.pos_weight_a,
        #                                                             pos_weight_s=self.pos_weight_s,
        #                                                             BCE_s=self.bce_s)

        # score, attr_error, stru_error = self.objective_function(
        #     ground_truth_node_feat=gt_attri.to_dense()[node_mask == 1, :],
        #     reconstructed_node_feat=pred_attri.to_dense(),
        #     ground_truth_node_struct=gt_adj.to_dense(
        #     )[node_mask == 1, :][:, node_mask == 1],
        #     reconstructed_node_struct=pred_adj.to_dense(),
        #     alpha=self.weight,
        #     pos_weight_a=self.pos_weight_a,
        #     pos_weight_s=self.pos_weight_s,
        #     BCE_s=self.bce_s,
        #     test=test)
        score, attr_error, stru_error = self.objective_function(
            ground_truth_node_feat=gt_attri.to_dense()[node_mask == 1, :],
            reconstructed_node_feat=pred_attri.to_dense(),
            ground_truth_node_struct=partial_mat,
            reconstructed_node_struct=pred_adj.to_dense(),
            alpha=self.weight,
            pos_weight_a=self.pos_weight_a,
            pos_weight_s=self.pos_weight_s,
            BCE_s=self.bce_s,
            test=test)

        return score, attr_error, stru_error

    def objective_function(self,
                           ground_truth_node_feat,
                           reconstructed_node_feat,
                           ground_truth_node_struct,
                           reconstructed_node_struct,
                           alpha=0.5,
                           pos_weight_a=0.5,
                           pos_weight_s=0.5,
                           BCE_s=False,
                           test=False):
        """
        Objective function of the proposed deep graph convolutional autoencoder, defined as
        :math:`\alpha \symbf{R_a} +  (1-\alpha) \symbf{R_s}`,
        where :math:`\alpha` is an important controlling parameter which balances the
        impacts of structure reconstruction and attribute reconstruction
        (it is a value betweeb 0 and 1 inclusive),
        and :math:`\symbf{R_a}` and :math:`\symbf{R_s}`
        are the reconstruction errors for attribute and structure, respectively.

        For attribute reconstruction error, we use mean squared error loss:
        :math:`\symbf{R_a} = \|\symbf{X}-\symbf{X}'\odot H\|`,
        where :math:`H=\begin{cases}1 - \eta &
        \text{if }x_{ij}=0\\ \eta & \text{if }x_{ij}>0\end{cases}`, and
        :math:`\eta` is the positive weight for feature.

        For structure reconstruction error, we use mean squared error loss by
        default: :math:`\symbf{R_s} = \|\symbf{S}-\symbf{S}'\odot
        \Theta\|`, where :math:`\Theta=\begin{cases}1 -
        \theta & \text{if }s_{ij}=0\\ \theta & \text{if }s_{ij}>0
        \end{cases}`, and :math:`\theta` is the positive weight for
        structure. Alternatively, we can use binary cross entropy loss
        for structure reconstruction: :math:`\symbf{R_s} =
        \text{BCE}(\symbf{S}, \symbf{S}' \odot \Theta)`.

        Parameters
        ----------
        ground_truth_nf : torch.Tensor
            Ground truth node feature
        reconstructed_nf : torch.Tensor
            Reconstructed node feature
        ground_truth_ns : torch.Tensor
            Ground truth node structure
        reconstructed_ns : torch.Tensor
            Reconstructed node structure
        alpha : float, optional
            Balancing weight :math:`\alpha` between 0 and 1 inclusive between node feature
            and graph structure. Default: ``0.5``.
        pos_weight_a : float, optional
            Positive weight for feature :math:`\eta`. Default: ``0.5``.
        pos_weight_s : float, optional
            Positive weight for structure :math:`\theta`. Default: ``0.5``.
        BCE_s : bool, optional
            Use binary cross entropy for structure reconstruction loss.

        Returns
        -------
        score : torch.tensor
            Outlier scores of shape :math:`N` with gradients.
        """

        assert 0 <= alpha <= 1, "weight must be a float between 0 and 1."
        assert 0 <= pos_weight_a <= 1 and 0 <= pos_weight_s <= 1, \
            "positive weight must be a float between 0 and 1."

        # attribute reconstruction loss
        # assert (torch.count_nonzero(torch.isnan(ground_truth_node_feat))
        #         ) == 0, "GT node feat contains nan"
        # assert (torch.count_nonzero(torch.isnan(reconstructed_node_feat))
        #         ) == 0, "Reconstructed node feat contains nan"

        diff_attr = torch.pow(ground_truth_node_feat -
                              reconstructed_node_feat, 2)

        if pos_weight_a != 0.5:
            diff_attr = torch.where(ground_truth_node_feat > 0,
                                    diff_attr * pos_weight_a,
                                    diff_attr * (1 - pos_weight_a))

        attr_error = torch.sqrt(torch.sum(diff_attr, 1))

        # structure reconstruction loss
        if BCE_s:
            diff_stru = F.binary_cross_entropy(
                reconstructed_node_struct, ground_truth_node_struct, reduction='none')
        else:
            # assert (torch.count_nonzero(torch.isnan(ground_truth_node_struct))
            #         ) == 0, "GT node struct contains nan"
            assert (torch.count_nonzero(torch.isnan(reconstructed_node_struct))
                    ) == 0, "Reconstructed node struct contains nan"

            diff_stru = ground_truth_node_struct - reconstructed_node_struct

            diff_stru = torch.pow(diff_stru, 2)

        if pos_weight_s != 0.5:
            diff_stru = torch.where(ground_truth_node_struct > 0,
                                    diff_stru * pos_weight_s,
                                    diff_stru * (1 - pos_weight_s))

        stru_error = torch.sqrt(torch.sum(diff_stru, 1))

        if test:
            score = alpha * attr_error.to("cpu") + \
                (1 - alpha) * stru_error.to("cpu")
        else:
            score = alpha * attr_error + \
                (1 - alpha) * stru_error
        # score = stru_error
        return score, attr_error, stru_error
