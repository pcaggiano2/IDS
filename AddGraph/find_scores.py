import os
import argparse
import torch
import numpy as np
from Graph_dataset import Graph_dataset
from copy import copy
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import os
from Graph_dataset import Graph_dataset
from torch.utils.data import DataLoader
from framework.model import *


def find_fp(y_true, y_pred):
    fp = 0
    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    return fp

def find_optimal_threshold(y_true, y_scores, threshold):
    threshold_list = []
    fp_list = []
    y_pred = [1.0 if score > threshold else 0.0 for score in y_scores]
    fp = find_fp(y_true, y_pred)
    fp_percentage = (fp / len(y_pred)) * 100

    print(f"Initial threshold: {threshold}, Initial FP percentage: {fp_percentage:.2f}%")
    
    threshold_list.append(threshold)
    fp_list.append(fp)
    final_threshold = copy(threshold)
    break_flag = False

    # Set a maximum iteration limit
    max_iterations = 10000  # Prevent infinite loops
    iteration = 0

    while not break_flag and iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}, Current Threshold: {final_threshold:.4f}, FP%: {fp_percentage:.2f}%")
        
        # Increase threshold if FP percentage is too high
        if fp_percentage > 1.1:
            final_threshold += final_threshold * 0.02  # Increase slightly
            print("FP percentage is above 1.1%; increasing threshold.")
        
        # Decrease threshold if FP percentage is too low
        elif fp_percentage < 0.9:
            final_threshold -= final_threshold * 0.05  # Decrease more aggressively
            print("FP percentage is below 0.9%; decreasing threshold.")
        
        # Exit loop if FP percentage is within the acceptable range
        else:
            print("FP percentage is within the target range.")
            break_flag = True
            continue

        # Update predictions and recalculate FP and percentage
        y_pred = [1.0 if score > final_threshold else 0.0 for score in y_scores]
        fp = find_fp(y_true, y_pred)
        fp_percentage = (fp / len(y_pred)) * 100

        threshold_list.append(final_threshold)
        fp_list.append(fp)

    if iteration >= max_iterations:
        print(f"Warning: Maximum iterations ({max_iterations}) reached. Final FP%: {fp_percentage:.2f}%")
    
    print(f"Final Threshold: {final_threshold:.4f}, Final FP%: {fp_percentage:.2f}%")
    return final_threshold, threshold_list, fp_list

def find_max_score(y_scores):
    max_score = np.max(y_scores)
    return max_score


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
        model_files = [f for f in files if f.startswith("AddGraph_model_")]
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

        Net1 = SpGAT(nfeat=args.hidden, nhid=args.nmid2, nout=args.hidden, dropout=args.dropout, alpha=args.alpha,
                    nheads=args.nb_heads)
        Net2 = HCA(hidden=args.hidden, dropout=args.dropout)
        Net3 = GRU(hidden=args.hidden, dropout=args.dropout)
        Net4 = Score(beta=args.beta, mui=args.mui, hidden=args.hidden, dropout=args.dropout)

        print('===> Loding models...')
        dir = os.path.join(directory_path, best_model_file)
        checkpoint = torch.load(dir)
        Net1.load_state_dict(checkpoint['Net1'])
        Net2.load_state_dict(checkpoint['Net2'])
        Net3.load_state_dict(checkpoint['Net3'])
        Net4.load_state_dict(checkpoint['Net4'])
        U_adj = checkpoint['U_adj']
        H_list = checkpoint['H_list']
        loss_a = checkpoint['loss_a']
        print(loss_a)

        print("Extracting ", best_model_file, "...")

        return Net1, Net2, Net3, Net4, U_adj, H_list, H_
    

    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden', type=int, default=50, help='Number of hidden units.')
    parser.add_argument('--nmid2', type=int, default=70, help='Number of nmid2 units.')
    parser.add_argument('--beta', type=float, default=3.0, help='Hyper-parameters in the score function.')
    parser.add_argument('--mui', type=float, default=0.5, help='Hyper-parameters in the score function.')
    parser.add_argument('--gama', type=float, default=0.6, help='Parameters in the score function.')
    parser.add_argument('--w', type=int, default=3, help='Hyper-parameters in the score function.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Number of graphs per epoch')

    parser.add_argument("--checkpoint_path",
                        type=str,
                        help="Folder from which take the model to evaluate")

    parser.add_argument("--device",
                        type=str,
                        #default="cuda",
                        default="mps",
                        help="GPU/CPU")

    parser.add_argument("--dataset_folder",
                        type=str,
                        help="Dataset folder from which take the graphs")

    parser.add_argument("--json_folder",
                        type=str,
                        help="Dataset folder in json format from which take the dataset split")

    parser.add_argument("--graph_type",
                        type=str,
                        help="Graph type to consider (similarity_graph/trajectory_graph/etdg_graph)")

    parser.add_argument("--threshold_path",
                        type=str,
                        help="Where to save a txt file with threshold found")

    parser.add_argument("--dataset",
                        type=str,
                        default="IoT23")

    parser.add_argument("--debug",
                        action='store_true')

    args = parser.parse_args()

    threshold_folder = os.path.join(args.threshold_path, args.graph_type)
    # create threshold folder if it does not exists
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger")
        debugpy.wait_for_client()

    # Dataloaders definition
    json_validation_set = os.path.join(args.json_folder, "val.json")
    graph_set = Graph_dataset(dataset_path=args.dataset_folder,
                              json_path=json_validation_set,
                              representation=args.graph_type)

    val_dataloader = DataLoader(
        graph_set, batch_size=args.batch_size, num_workers=0)

    device = None
    if args.device == 'cuda':
        device = torch.device(args.device)
    elif args.device == 'mps':
        device = torch.device(args.device)

    Net1, Net2, Net3, Net4, U_adj, H_list, H_ = get_best_model(args, os.path.join(
                                    args.checkpoint_path, args.graph_type))
    # check if GPU is available
    device = torch.device(args.device)

    print("\n Starting to find scores...")

    Net1.to(device)
    Net2.to(device)
    Net3.to(device)
    Net4.to(device)
    Net1.eval()
    Net2.eval()
    Net3.eval()
    Net4.eval()
    score_list = []
    labels_list = []
    with torch.no_grad():
        #Versione solo con Net4
        for snapshot, labels in zip(val_dataloader):
            H = H_list[-1]
            snapshot_score_list = []
       
            for src, dst in snapshot:
                src = src.to(device)
                dst = dst.to(device)
                score = Net4(hi=H[src - 1], hj=H[dst - 1])
                snapshot_score_list.append(score.detach().cpu().numpy())
            
            score_list.extend(snapshot_score_list)
            labels_list.extend(labels)

    # print(score_list)
    print("\n Saving scores...")
    np.save(os.path.join(threshold_folder, "score_list.npy"), score_list)
    np.save(os.path.join(threshold_folder, "label_list.npy"), labels_list)

    # computing threshold
    print("\n Computing threshold...")
    threshold_folder = os.path.join(args.threshold_path, args.graph_type)
    # create threshold folder if it does not exists
    if not os.path.exists(threshold_folder):
        os.makedirs(threshold_folder)

    y_scores = np.load(os.path.join(threshold_folder, "score_list.npy"))
    y_true = np.load(os.path.join(threshold_folder, "label_list.npy"))
    max_score = find_max_score(y_scores)
    print("Max score:", max_score)

    threshold = max_score/2
    print("Starting threshold:", threshold)
    optimal_threshold, threshold_list, fp_list = find_optimal_threshold(
        y_true, y_scores, threshold)

    print("Optimal threshold to use in the next step: ", optimal_threshold)
    threshold_file_path = os.path.join(
        threshold_folder, "optimal_threshold.txt")
    with open(threshold_file_path, "w") as f:
        f.write(str(optimal_threshold))
