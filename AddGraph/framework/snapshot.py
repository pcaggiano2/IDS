# _*_ coding:utf-8 _*_
import numpy as np

from framework.anomaly_generation import anomaly_generation
from framework.load_uci_messages import load_uci_messages
from framework.negative_sample import negative_sample
from framework.model import *
from tqdm import tqdm

def snapshot(data_path, sample_rate, ini_graph_percent, anomaly_percent, snapshots_):
    """
    Generates snapshots from the dataset by loading data, creating anomalies, 
    and splitting it into training and testing snapshots.

    Args:
        data_path (str): Path to the input data file.
        sample_rate (float): Rate at which to sample data.
        ini_graph_percent (float): Percentage of the initial graph to consider.
        anomaly_percent (float): Percentage of anomalies to introduce.
        snapshots_ (int): Number of edges in each snapshot.

    Returns:
        tuple: Training and testing snapshots along with their counts and other metadata.
    """
    # Load the dataset
    data, n, m = load_uci_messages(data_path, sample_rate)
    
    # Generate anomalies and split data into training and synthetic test sets
    n_train, train, synthetic_test = anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m)

    # Prepare training snapshots
    l_train = np.size(train, 0)
    snapshots_train = []
    current = 0
    while (l_train - current) >= snapshots_:
        snapshots_train.append(train[current:current + snapshots_])
        current += snapshots_
    snapshots_train.append(train[current:])  # Add remaining edges as the last snapshot
    print("Train data: number of snapshots: %d, edges in each snapshot: %d" % (len(snapshots_train), snapshots_))

    # Prepare testing snapshots
    l_test = np.size(synthetic_test, 0)
    snapshots_test = []
    current = 0
    while (l_test - current) >= snapshots_:
        snapshots_test.append(synthetic_test[current:current + snapshots_])
        current += snapshots_
    snapshots_test.append(synthetic_test[current:])  # Add remaining edges as the last snapshot
    print("Test data: number of snapshots: %d, edges in each snapshot: %d" % (len(snapshots_test), snapshots_))

    return snapshots_train, len(snapshots_train), snapshots_test, len(snapshots_test), n, n_train


def normalize_adj(adj):
    """Row-normalize sparse matrix.

    Args:
        adj (torch.Tensor): Adjacency matrix to be normalized.

    Returns:
        torch.Tensor: Row-normalized adjacency matrix.
    """
    D = adj.sum(1)  # Sum along rows to get the degree of each node
    r_inv_sqrt = D.pow(-0.5)  # Compute the inverse square root of the degree
    r_inv_sqrt[torch.eq(r_inv_sqrt, float('inf'))] = 0.  # Replace infinities with zeros
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)  # Create diagonal matrix from inverse square root
    # Normalize the adjacency matrix
    return torch.mm(torch.mm(adj, r_mat_inv_sqrt).t(), r_mat_inv_sqrt)

if __name__ == "__main__":
    # Set hyperparameters for the model
    nfeat = nhid = hidden = 100
    nmid1 = 200
    nmid2 = 150

    # Uncomment and define your models as needed
    # Net1 = GCN(nfeat=nfeat, nmid1=nmid1, nmid2=nmid2, nhid=nhid, dropout=0)
    # Net2 = HCA(hidden=hidden, dropout=0)
    # Net3 = GRU(hidden=hidden, dropout=0)
    # Net4 = Score(beta=1.0, mui=0.3, hidden=hidden, dropout=0)

    data_path = '../opsahl-ucsocial/out.opsahl-ucsocial'
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # Generate snapshots and save them for later use
    snapshots_train, l_train, snapshots_test, l_test, nodes, adj = snapshot(data_path, 0.3, 0.5, 0.2, 1000)
    np.savez("snapshot_10a.npz", snapshots_train=snapshots_train, l_train=l_train, 
             snapshots_test=snapshots_test, l_test=l_test, nodes=nodes, m=adj)

    # Load the saved snapshots
    snapshots = np.load("snapshot_10a.npz", allow_pickle=True)
    A_adj = snapshots['m']

    # Example usage of the NegativeSample class
    # H = torch.rand(nodes, hidden)  # Random initialization of node embeddings
    # H_ = torch.zeros(3, nodes, hidden)  # Placeholder for future embeddings
    # adj = np.zeros((nodes, nodes))  # Placeholder adjacency matrix
    # snapshot = snapshots_train[0]  # Example snapshot
    # n_data, n_loss, adj = NegativeSample(adj, snapshots_train[0], H, Net4)  # Example call
    # adjn = normalize_adj(adj + np.eye(adj.shape[0]))  # Normalize adjacency
    # adj_ = torch.from_numpy(adjn)  # Convert to tensor

    # Print sample outputs or scores
    # for i in tqdm(range(len(n_data))):
    #     normal = snapshot[i]
    #     anormal = n_data[i]
    #     score1 = Net4(hi=Hn[normal[0]-1], hj=Hn[normal[1]-1])  # Calculate score for normal edge
    #     score2 = Net4(hi=Hn[anormal[0] - 1], hj=Hn[anormal[1] - 1])  # Calculate score for anomalous edge
    #     print(i + 1)
    #     print(score1, ',', score2)
    #     print('\n')

    # Example loss calculation (commented out)
    # loss = Net1.loss() + Net2.loss() + Net3.loss()
    # print(loss[0])
    # print(loss.grad)

    # Additional examples for negative sampling
    # n1_data, adj = NegativeSample(adj, snapshots_train[1])
    # print(n1_data, adj)
    # print(snapshots_test[10])
    # print(np.array(snapshots_test[0][:, 0:2]))
