import os
import json
import numpy as np
import pickle as pk
import argparse

# Provided functions
def create_path(capture, representation, set_type, graph_name):
    if set_type == "train" or set_type == "val" or set_type == "test_benign":
        path = os.path.join(capture, representation, "full_benign", graph_name)
    elif set_type == "test_mixed":
        path = os.path.join(capture, representation, "mixed", graph_name)
    elif set_type == "test_malicious":
        path = os.path.join(capture, representation, "full_malicious", graph_name)
    return path

def get_list_files(json_path, representation):
    file_paths = []
    # Read JSON object
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    # Iterate over JSON to create the original data path
    set_type = os.path.splitext(os.path.basename(json_path))[0]
    for capture, graph_name_list in json_data.items():
        for graph_name in graph_name_list:
            #for graph in graph_name:
                #print(graph_name)
            file_path = create_path(capture, representation, set_type, graph_name)
            file_paths.append(file_path)
    file_paths.sort()
    return file_paths

# New function to compute min and max
def compute_min_max(json_path, representation, dataset_path):
    """
    Computes the min and max values for node features using the list of graphs provided in the JSON file.

    Args:
        json_path (str): Path to the JSON file specifying the dataset structure.
        representation (str): Graph representation type.
        dataset_path (str): Base path to the dataset directory.

    Returns:
        tuple: (min_vals, max_vals) arrays for the minimum and maximum values.
    """
    min_vals = None
    max_vals = None

    # Get the list of graph file paths
    graph_file_paths = get_list_files(json_path, representation)
    for file_path in graph_file_paths:
        full_path = os.path.join(dataset_path, file_path)

        # Load the graph
        with open(full_path, "rb") as f:
            graph = pk.load(f)
            node_features = graph['node_features']

            # Update min and max values
            if min_vals is None or max_vals is None:
                min_vals = np.min(node_features, axis=0)
                max_vals = np.max(node_features, axis=0)
            else:
                min_vals = np.minimum(min_vals, np.min(node_features, axis=0))
                max_vals = np.maximum(max_vals, np.max(node_features, axis=0))

    return min_vals, max_vals

def save_min_max(folder_path, min_vals, max_vals):
    """
    Save the computed min and max values as `.npz` files.

    Args:
        folder_path (str): Path to save the `min.npz` and `max.npz` files.
        min_vals (np.array): Array of minimum values for each feature.
        max_vals (np.array): Array of maximum values for each feature.
    """
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

    np.savez(os.path.join(folder_path, "min.npz"), min_vals)
    np.savez(os.path.join(folder_path, "max.npz"), max_vals)

    print(f"Min and max values saved to {folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute min and max values for node features.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Base path to the dataset directory.")
    parser.add_argument("--output_folder", type=str, required=True, help="Where to save min.npz and max.npz")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file specifying the dataset structure.")
    parser.add_argument("--representation", type=str, required=True, help="Graph representation type.")
    
    args = parser.parse_args()

    json_path = args.json_path  # Path to your JSON file
    representation = args.representation # Replace with your representation type
    dataset_path = args.dataset_path  # Base dataset directory
    output_folder = args.output_folder  # Where to save min.npz and max.npz

    # Step 1: Compute min and max
    min_vals, max_vals = compute_min_max(json_path, representation, dataset_path)

    # Step 2: Save min and max to files
    save_min_max(output_folder, min_vals, max_vals)

    print("Min values:", min_vals)
    print("Max values:", max_vals)



