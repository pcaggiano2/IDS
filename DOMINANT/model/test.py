import torch
import os
import csv
import time
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from DOMINANT import DOMINANT
from Graph_dataset import Graph_dataset
from functional import objective_function
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def get_best_model(args, directory_path):
    """
    Accesses the specified directory and selects the file with the highest numerical index
    in the file name. Assumes that the files have a common prefix followed by a numerical index.

    Parameters
    ----------
    directory_path (str): The path of the directory containing the model files.

    Returns
    ----------
    str: The name of the file with the highest numerical index, or None if the directory is empty or contains no valid files.
    """
    # List to store the numerical indices of the files
    model_indices = []
    try:
        # List all files in the directory
        files = os.listdir(directory_path)
        # Filter files that start with the prefix "DOMINANT_model_"
        model_files = [f for f in files if f.startswith("DOMINANT_model_")]
        # Extract numerical indices from the file names
        for model_file in model_files:
            try:
                index = int(model_file.split("_")[-1])
                model_indices.append((model_file, index))
            except ValueError:
                # Ignore files that do not have a valid numerical index
                continue
        # If there are no valid files, return None
        if not model_indices:
            return None
        # Sort the list of tuples based on the numerical index and select the file with the highest index
        best_model_file = sorted(
            model_indices, key=lambda x: x[1], reverse=True)[0][0]

        if args.graph_type == "tdg_graph" or args.graph_type == "sim_graph":
            in_dim = 57
        if args.graph_type == "etdg_graph":
            in_dim = 15#72

        model = DOMINANT(in_dim=in_dim,
                         hid_dim=args.hidden_dim,
                         encoder_layers=args.encoder_layers,
                         decoder_layers=args.decoder_layers,
                         dropout=args.dropout)

        model.load_state_dict(torch.load(
            os.path.join(directory_path, best_model_file), map_location=torch.device('cpu')))
        print("Extracting ", best_model_file, "...")

        return model

    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return None


def import_optimal_threshold(threshold_folder):
    """
    Reads the content of the file named "optimal_threshold.txt" found in the specified directory.

    Parameters
    ----------
    directory_path : str
        The path of the directory from which to fetch the file.

    Returns
    ----------
    float
        The content of the file as a float, or None if the file is not found or an error occurs.
    """

    try:
        # Build the full path of the file named "optimal_threshold.txt"
        file_path = os.path.join(threshold_folder, "optimal_threshold.txt")

        # Check if the file exists
        if not os.path.isfile(file_path):
            print(
                f"The file optimal_threshold.txt does not exist in the directory {threshold_folder}.")
            return None

        # Open and read the file
        with open(file_path, 'r') as file:
            threshold = file.read()

        return float(threshold)

    except FileNotFoundError:
        print(
            f"The directory {threshold_folder} or the file optimal_threshold.txt does not exist.")
        return None
    except IOError:
        print(
            f"An error occurred while reading the file optimal_threshold.txt in {threshold_folder}.")
        return None
    except ValueError:
        print(
            f"The content of optimal_threshold.txt in {threshold_folder} could not be converted to float.")
        return None


def process_graph(data):
    """
    Obtain the dense adjacency matrix of the graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input graph.
    """
    data.s = to_dense_adj(data.edge_index)[0]


def compute_node_anomaly_score(x, x_, s, s_, alpha):
    """
    This function computes the anomaly score for each node in a given graph/batch.
    """

    score, _, _ = objective_function(x,
                                     x_,
                                     s,
                                     s_,
                                     alpha)

    return score.detach().cpu()


def compute_evaluation_metrics(threshold, y_scores, y_true):
    """
    Compute evaluation metrics for binary classification tasks.

    Parameters:
    - threshold (float): The decision threshold for classifying anomalies. 
                         Anomaly scores greater than this value are classified as anomalous (1), 
                         and scores less than or equal to this value are classified as normal (0).

    - y_scores (list of float): An array of anomaly scores, typically between 0 and 1, 
                                where higher values indicate higher likelihood of being an anomaly.

    - y_true (list of int): An array of true labels where 1 indicates an anomalous instance 
                            and 0 indicates a normal instance.

    Returns:
    - accuracy (float): The proportion of correctly classified instances.
    - precision (float): The proportion of true positive instances among the instances that are 
                         predicted as positive.
    - recall (float): The proportion of true positive instances among the instances that are 
                      actually positive.
    - f_score (float): The weighted harmonic mean of precision and recall.

    Note:
    The function assumes that the input arrays y_scores and y_true have the same length.
    """
    y_pred = [1. if score > threshold else 0. for score in y_scores]
    if len(y_pred) != len(y_true):
        print("Error")
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    try:
        for i in range(len(y_pred)):
            if y_pred[i] == 1 and y_true[i] == 1:
                tp += 1
            if y_pred[i] == 0 and y_true[i] == 0:
                tn += 1
            if y_pred[i] == 1 and y_true[i] == 0:
                fp += 1
            if y_pred[i] == 0 and y_true[i] == 1:
                fn += 1

        print("\nTP:", tp)
        print("\nTN:", tn)
        print("\nFP:", fp)
        print("\nFN:", fn)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * (precision * recall) / (precision + recall)
    except:
        precision = 0
        recall = 0
        f_score = 0

    return accuracy, precision, recall, f_score, tp, tn, fp, fn


def evaluate(args, model, dataset):
    """
    This function evaluates the model on the IoT23 dataset and writes the results to a CSV file.

    Parameters
    ----------
    args: Command-line arguments or a configuration object containing parameters like dataset folder, graph type, etc.
    model: The model to be evaluated.

    Returns
    ----------
    None. Writes evaluation metrics to a CSV file.
    """
    # paths definition
    json_set = os.path.join(args.json_folder, dataset)
    threshold_folder = os.path.join(args.threshold_path, args.graph_type)

    opt_threshold = import_optimal_threshold(threshold_folder)
    print("Threshold to use: ", opt_threshold)

    # Dataloaders definition
    graph_set = Graph_dataset(dataset_path=args.dataset_folder,
                              json_path=json_set,
                              representation=args.graph_type,
                              normalize=args.normalize,
                              min_max=args.min_max)
    test_dataloader = DataLoader(graph_set, num_workers=0)

    # check if GPU is available
    device = torch.device(args.device)

    total_time = 0.0
    score_list = []
    labels_list = []
    model.to(device)
    model.eval()
    print("Testing {} set...".format(dataset))
    with torch.no_grad():
        
        for batch in tqdm(test_dataloader):
            try:
                process_graph(batch)
                x = batch.x.to(device)
                s = batch.s.to(device)
                edge_index = batch.edge_index.to(device)
                start_time = time.time()  # seconds
                x_, s_ = model(x, edge_index)
                anomaly_score, _, _ = objective_function(x, x_, s, s_, args.alpha)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time

                score_list.extend(anomaly_score.detach().cpu().numpy().tolist())
                labels_list.extend(batch.y.numpy().tolist())
            except Exception as e:
                print(f"Graph too large, next...")
                pass

    average_pred_time = total_time / len(test_dataloader)

    accuracy, precision, recall, fscore, tp, tn, fp, fn = compute_evaluation_metrics(
        opt_threshold, score_list, labels_list)
    test_writer.writerow([dataset, accuracy, precision,
                         recall, fscore, tp, tn, fp, fn, average_pred_time*1000])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_dim',
                        type=int,
                        default=64,
                        help='Dimension of hidden embedding (default: 28)')

    parser.add_argument('--encoder_layers',
                        type=int,
                        default=3,
                        help='Number of encoder layers')

    parser.add_argument('--decoder_layers',
                        type=int,
                        default=2,
                        help='Number of decoder layers')

    parser.add_argument('--lr',
                        type=float,
                        default=5e-4,
                        help='Learning rate')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate')

    parser.add_argument('--alpha',
                        type=float,
                        default=0.7,
                        help='Balance parameter')

    parser.add_argument('--device',
                        type=str,
                        #default='cuda',
                        default='mps',
                        help='GPU = cuda/CPU = cpu')

    parser.add_argument("--graph_type",
                        type=str,
                        help="Graph type to consider (similarity_graph/tdg_graph)")

    parser.add_argument("--checkpoint_path",
                        type=str,
                        help="Folder from which take the model to evaluate")

    parser.add_argument("--dataset_folder",
                        type=str,
                        help="Dataset folder from which take the graphs")

    parser.add_argument("--json_folder",
                        type=str,
                        help="Dataset folder in json format from which take the dataset split")

    parser.add_argument("--result_path",
                        type=str,
                        help="Folder where to save test results")

    parser.add_argument("--threshold_path",
                        type=str,
                        help="Folder with optimal thresholds")

    parser.add_argument("--normalize",
                        type=int,
                        default=-1)

    parser.add_argument("--dataset",
                        type=str)

    parser.add_argument("--min_max",
                        type=str,
                        help="Path to min and max numpy arrays")

    parser.add_argument("--debug",
                        action='store_true')

    parser.add_argument("--wandb_log",
                        action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger")
        debugpy.wait_for_client()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    csv_results_folder = os.path.join(args.result_path, args.graph_type)
    # create csv results folder if it does not exists
    if not os.path.exists(csv_results_folder):
        os.makedirs(csv_results_folder)
    test_result_file = open(os.path.join(
        csv_results_folder, f'result_{args.dataset}.csv'), "w", newline='')
    test_writer = csv.writer(test_result_file)
    test_writer.writerow(['Set', 'Accuracy', 'Precision',
                         'Recall', 'F-Score', "TP", "TN", "FP", "FN", 'Average pred time(ms)'])

    model = get_best_model(args, os.path.join(
        args.checkpoint_path, args.graph_type))

    if args.dataset == "IoT23":
        datasets = ["val.json",
                    "test_malicious.json", 
                    "test_mixed.json"]
    elif args.dataset == "IoT_traces":
        datasets = ["test_benign.json"]
    elif args.dataset == "IoTID20":
        datasets = ["test_benign.json", "test_mixed.json"]
    elif args.dataset == "Bot-IoT":
        datasets = ["test_benign.json", "test_malicious.json","test_mixed.json"]

    for dataset in datasets:
        print(
            f"\n Starting the evaluation for {args.dataset}, {args.graph_type}, ...")
        evaluate(args, model, dataset)

    test_result_file.close()
