# _*_ coding:utf-8 _*_
# @author: Jiajie Lin
# @file: anomaly_generation.py
# @time: 2020/03/08

import numpy as np
from framework.load_uci_messages import load_uci_messages

def anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m):
    """
    Generate synthetic anomalies and split the graph into training and testing sets.
    
    Args:
    - ini_graph_percent (float): Percentage of edges used in the training set.
    - anomaly_percent (float): Percentage of edges in the test set that should be anomalous.
    - data (ndarray): Edge list of the entire graph, each row is an edge (nodeID, nodeID).
    - n (int): Number of nodes in the graph.
    - m (int): Total number of edges in the graph.
    
    Returns:
    - n_train (int): Number of unique nodes in the training set.
    - train (ndarray): Training set edge list (subset of the original graph).
    - synthetic_test (ndarray): Test set edge list with injected anomalies, each row contains (nodeID, nodeID, label),
      where label == 0 means normal edge, label == 1 means anomalous edge.
    """
    print('Generating anomalous dataset...\n')
    print(f'Initial network edge percent: {ini_graph_percent * 100}%')
    print(f'Initial anomaly percent: {anomaly_percent * 100}%\n')

    # Number of edges to use in the training set
    train_num = int(np.floor(ini_graph_percent * m))
    
    # region Training and Test Edges
    # Use the first 'train_num' edges as the training set
    train = data[:train_num, :]
    train_ = np.unique(train)  # Get unique nodes in the training set
    n_train = len(train_)

    # Create an adjacency matrix for the training set
    adj = np.zeros((n, n))
    for edge in train:
        adj[edge[0] - 1][edge[1] - 1] += 1  # 1-based to 0-based index
    
    # Use the remaining edges as the test set
    test = data[train_num:, :]
    
    # region Generate Fake Anomalous Edges
    # Number of anomalies to inject into the test set
    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    
    # Initialize labels for test edges (0 for normal, 1 for anomalous)
    idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    
    # Randomly select positions in the test set to inject anomalies
    anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)
    idx_test[anomaly_pos] = 1  # Set those positions as anomalies
    
    # region Prepare Synthetic Test Edges
    # Inject anomalies into the test set
    idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
    idx_normal = np.nonzero(idx_test.squeeze() == 0)
    
    # Initialize the test set with normal edges
    test_aedge = np.zeros([np.size(idx_test, 0), 2], dtype=np.int32)
    test_aedge[idx_normal] = test
    
    # Process and inject the anomalies into the test set
    test_edge = processEdges(idx_anomalies[0], test_aedge, adj)
    
    # Combine test edges with their corresponding labels (normal or anomalous)
    synthetic_test = np.concatenate((test_edge, idx_test), axis=1)
    
    return n_train, train, synthetic_test

def processEdges(idx_anomalies, test_aedge, adj):
    """
    Generate and inject anomalous edges into the test set.
    
    Args:
    - idx_anomalies (ndarray): Indices in the test set where anomalies should be injected.
    - test_aedge (ndarray): Test edge array.
    - adj (ndarray): Adjacency matrix of the training graph.
    
    Returns:
    - test_aedge (ndarray): Test edge array with injected anomalous edges.
    """
    for idx in idx_anomalies:
        flag = 0
        th = np.max(test_aedge[0:idx, :])
        # Randomly select two nodes to form a fake edge
        idx_1, idx_2 = np.random.choice(th, 2, replace=False) + 1
        
        # Ensure the edge doesn't already exist in the graph
        while adj[idx_1 - 1][idx_2 - 1] != 0:
            idx_1, idx_2 = np.random.choice(th, 2, replace=False) + 1
        
        # Ensure no duplicates or self-loops
        while flag == 0:
            for edge in test_aedge[0:idx, :]:
                if (idx_1 == edge[0] and idx_2 == edge[1]) or (idx_1 == edge[1] and idx_2 == edge[0]):
                    flag = 1
                    break
            if flag == 0:
                # Inject the anomalous edge
                test_aedge[idx, 0] = idx_1
                test_aedge[idx, 1] = idx_2
                break
            else:
                idx_1, idx_2 = np.random.choice(th, 2, replace=False) + 1
                flag = 0
    return test_aedge

def edgeList2Adj(data):
    """
    Convert an edge list to a symmetric adjacency matrix.
    
    Args:
    - data (ndarray): Edge list, each row is an edge (nodeID, nodeID).
    
    Returns:
    - matrix (ndarray): Symmetric adjacency matrix.
    """
    data = tuple(map(tuple, data))  # Convert to tuple for efficient processing
    n = max(max(user, item) for user, item in data)  # Get size of the matrix
    matrix = np.zeros((n, n))  # Initialize the adjacency matrix
    
    # Populate the adjacency matrix
    for user, item in data:
        matrix[user - 1][item - 1] = 1  # 1-based to 0-based index
        matrix[item - 1][user - 1] = 1  # Ensure the matrix is symmetric
    
    return matrix


if __name__ == "__main__":
    # Example usage with UCI Messages dataset
    data_path = '../opsahl-ucsocial/out.opsahl-ucsocial'
    data, n, m = load_uci_messages(data_path, 0.3)  # Load the dataset
    edges = data
    vertices = np.unique(edges)
    
    # Generate anomaly-injected train and test sets
    train, synthetic_test = anomaly_generation(0.5, 0.8, edges, n, m)
    
    print(train.shape)
    print('\n')
    print(synthetic_test[:20, :])
    print(synthetic_test[-20:-1, :])
    print(synthetic_test.shape)
