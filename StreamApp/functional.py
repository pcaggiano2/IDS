import torch
import torch.nn.functional as F


def objective_function(ground_truth_node_feat,
                       reconstructed_node_feat,
                       ground_truth_node_struct,
                       reconstructed_node_struct,
                       alpha=0.5,
                       pos_weight_a=0.5,
                       pos_weight_s=0.5,
                       BCE_s=False):
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
    diff_attr = torch.pow(ground_truth_node_feat - reconstructed_node_feat, 2)

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
        diff_stru = torch.pow(ground_truth_node_struct -
                              reconstructed_node_struct, 2)

    if pos_weight_s != 0.5:
        diff_stru = torch.where(ground_truth_node_struct > 0,
                                diff_stru * pos_weight_s,
                                diff_stru * (1 - pos_weight_s))

    stru_error = torch.sqrt(torch.sum(diff_stru, 1))

    # assert torch.sum(torch.isnan(stru_error) == True) == 0, "Loss contains nan"

    score = alpha * attr_error + (1 - alpha) * stru_error
    # assert torch.sum(score < 0) == 0, "score is < 0"

    return score, attr_error, stru_error
