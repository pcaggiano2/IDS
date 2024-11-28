# _*_ coding:utf-8 _*_
# @author: Jiajie Lin
# @file: load_uci_messages.py
# @time: 2020/03/08

import numpy as np

def load_uci_messages(data_path, sample_rate):
    """
    Load and preprocess the UCI Message dataset.
    
    Args:
    - data_path (str): Path to the dataset file.
    - sample_rate (float): Rate at which to subsample the edges (0 < sample_rate <= 1).
    
    Returns:
    - data (ndarray): The processed edge list where each row represents an edge.
    - n (int): Number of unique nodes in the graph.
    - m_ (int): Number of edges after subsampling.
    """
    # Load edges from the dataset (assumes data is in EdgeList format)
    oedges = np.loadtxt(data_path, dtype=int, comments='%', usecols=(0, 1))

    # Get the total number of edges
    m = len(oedges)

    # Subsample the edges based on the provided sample rate
    m_ = int(np.floor(m * sample_rate))
    oedges = oedges[0:m_, :]  # Keep only a fraction of edges

    # Re-assign unique node IDs
    unique_id = np.unique(oedges)  # Extract unique node IDs from edges
    n = len(unique_id)  # Number of unique nodes

    # Map edges to a new set of continuous IDs (from 1 to n)
    _, digg = ismember(oedges, unique_id)

    # Return the processed edge list, number of nodes, and the subsampled number of edges
    return digg, n, m_

def ismember(a, b_vec):
    """
    Mimics MATLAB's ismember function.
    Maps the values in array 'a' to indices in 'b_vec', returning a flag array and mapped indices.
    
    Args:
    - a (ndarray): Array to map (e.g., edges in the graph).
    - b_vec (ndarray): Array of unique values to map against (e.g., unique node IDs).
    
    Returns:
    - flag (ndarray): Boolean array indicating if each element in 'a' is found in 'b_vec'.
    - content (ndarray): Array of indices that map 'a' elements to their positions in 'b_vec'.
    """
    shape_a = a.shape
    a_vec = a.flatten()  # Flatten the input array

    # Boolean index where a_vec is found in b_vec
    bool_ind = np.isin(a_vec, b_vec)
    common = a_vec[bool_ind]

    # Get unique values and their inverse indices from the common elements
    common_unique, common_inv = np.unique(common, return_inverse=True)

    # Get the indices of unique elements in b_vec
    b_unique, b_ind = np.unique(b_vec, return_index=True)
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]

    # Reshape the flag and content to match the original shape of 'a'
    flag = bool_ind.reshape(shape_a)
    content = (common_ind[common_inv]).reshape(shape_a) + 1  # +1 for 1-based index

    return flag, content

if __name__ == "__main__":
    # Example usage for testing the loading function
    data_path = '../munmun_digg_reply/out.munmun_digg_reply'
    data, n, m = load_uci_messages(data_path, 0.25 * 0.5)  # Load dataset with a subsample rate
    print(data.shape, n, m)  # Print shape of the processed data, number of nodes, and edges
