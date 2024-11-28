# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: test.py
# @time: 2020/03/20
import numpy as np
import torch
import torch.nn as nn
import argparse
from framework.model import *
from framework.negative_sample import update_adj
import os
import csv
from Graph_dataset import Graph_dataset
from torch.utils.data import DataLoader
import time

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
    
def compute_evaluation_metrics(y_scores, y_true, threshold=0.5):
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

def test(args, dataset):
    #Load model
    Net1, Net2, Net3, Net4, U_adj, H_list, H_ = get_best_model(args, os.path.join(
                                                            args.checkpoint_path, args.graph_type))
    # check if GPU is available
    device = torch.device(args.device)

    json_set = os.path.join(args.json_folder, dataset)
    json_training_set = os.path.join(args.json_folder, "train.json")
    # threshold_folder = os.path.join(args.threshold_path, args.graph_type)
    # opt_threshold = import_optimal_threshold(threshold_folder)
    # print("Threshold to use: ", opt_threshold)   

    test_set = Graph_dataset(dataset_path=args.dataset_folder,
                              json_path=json_set,
                              representation=args.graph_type)
    test_dataloader = DataLoader(test_set, num_workers=0)

    train_set = Graph_dataset(dataset_path=args.dataset_folder,
                              json_path=json_training_set,
                              representation=args.graph_type)
    
    n_train = train_set.len()
    nodes = train_set.n_nodes
    nn.init.sparse_(H_list[-1][n_train: , :].t(), sparsity=0.9)
    H_ = torch.zeros((args.w, nodes, args.hidden))


    total_time = 0.0
    score_list = []
    labels_list = []
    Net1.to(device)
    Net2.to(device)
    Net3.to(device)
    Net4.to(device)
    Net1.eval()
    Net2.eval()
    Net3.eval()
    Net4.eval()
    print("Testing {} set...".format(dataset))
    with torch.no_grad():
        # Versione solo con Net4
        for snapshot, labels in zip(test_dataloader):
            H = H_list[-1]
            snapshot_score_list = []
            start_time = time.time()  # seconds
            for src, dst in snapshot:
                src = src.to(device)
                dst = dst.to(device)
                score = Net4(hi=H[src - 1], hj=H[dst - 1])
                snapshot_score_list.append(score.detach().cpu().numpy())
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time

            score_list.extend(snapshot_score_list)
            labels_list.extend(labels)
        
        average_pred_time = total_time / len(test_dataloader)

        accuracy, precision, recall, fscore, tp, tn, fp, fn = compute_evaluation_metrics(
            score_list, labels_list) #Da inserire la treshold 
        test_writer.writerow([dataset, accuracy, precision,
                         recall, fscore, tp, tn, fp, fn, average_pred_time*1000])
        
        # # Versione tutta la pipeline
        # for snapshot, labels in zip(test_dataloader):
        #     H = H_list[-1]
        #     H_ = torch.zeros((args.w, nodes, args.hidden))

        #     # Update adjacency matrix and model inputs
        #     U_adj, adj, Adj = update_adj(U_adj=U_adj, snapshot=snapshot, nodes=nodes)
            
        #     H = H.to(device)
        #     Adj = Adj.to(device)
        #     snapshot = snapshot.to(device)

        #     # Forward pass through the models
        #     current = Net1(x=H, Adj=Adj, adj=adj)
        #     short = Net2(C=H_)
        #     Hn = Net3(current=current, short=short)
        #     H_list = torch.cat([H_list, Hn.unsqueeze(0)], dim=0)



    #############################################
    # for i in range(l_test):
    #     # global n_train, H_list, H_, U_adj
    #     data = snapshots_test[i]
    #     # snapshot = torch.from_numpy(data[:,(0,1)])
    #     snapshot_ = data[:, (0, 1)]
    #     # n_test_ = np.unique(snapshot_)
    #     # n_test = np.max(n_test_)
    #     # nn.init.sparse_(H_list[-1][n_train:n_test, :].t(), sparsity=0.9)
    #     # n_train = n_test
    #     label = data[:, 2]
    #     H = H_list[-1]
    #     prob = []
    #     for edge in snapshot_:
    #         m = edge[0]
    #         n = edge[1]
    #         score = Net4(hi=H[m - 1], hj=H[n - 1])
    #         prob.append(score.detach().numpy())
    #     prob = np.array(prob)
    #     auc = AUC(label=label, pre=prob)
    #     snapshot, acc = edge_p(edge=snapshot_, prob=prob, label=label, a=args.anomaly_percent)
    #     print('In snapshot {}'.format(i), 'the AUC results: {}.'.format(auc), 'the Acc results: {}.'.format(acc))

       
        # snapshot = torch.from_numpy(snapshot)
        # for j in range(args.w):
        #     H_[j] = H_list[-args.w + j]
        # # adj, Adj = update_adj(adj=adj, snapshot=snapshot, nodes=nodes)
        # U_adj, adj, Adj = update_adj(U_adj=U_adj, snapshot=snapshot, nodes=nodes)
        # # Adjn = normalize_adj(Adj + torch.eye(Adj.shape[0]))
        # H, H_, Adj, snapshot = Variable(H), Variable(H_), Variable(Adj), Variable(snapshot)
        # current = Net1(x=H, Adj=Adj, adj=adj)
        # short = Net2(C=H_)
        # Hn = Net3(current=current, short=short)
        # H_list = torch.cat([H_list, Hn.unsqueeze(0)], dim=0)


def edge_p(edge, prob, label, a):
    rank = np.argsort(prob)
    l = len(prob)
    normal_index = rank[:-int(np.floor(l * a))]
    anomaly_index = rank[-int(np.floor(l * a)):]
    neg_l = np.where(label == 1)
    edge_p = np.delete(edge, neg_l, axis=0)
    count = 0
    for i in (normal_index):
        if label[i] == 0:
            count = count + 1
    for j in (anomaly_index):
        if label[j] == 1:
            count = count + 1
    return edge_p, (count / l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--hidden', type=int, default=50, help='Number of hidden units.')
    parser.add_argument('--nmid2', type=int, default=70, help='Number of nmid2 units.')
    parser.add_argument('--beta', type=float, default=3.0, help='Hyper-parameters in the score function.')
    parser.add_argument('--mui', type=float, default=0.5, help='Hyper-parameters in the score function.')
    parser.add_argument('--gama', type=float, default=0.6, help='Parameters in the score function.')
    parser.add_argument('--w', type=int, default=3, help='Hyper-parameters in the score function.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
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
    
    parser.add_argument("--dataset",
                        type=str)
    
    args = parser.parse_args()

    if args.device == "cuda":
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        print('OK')
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

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

    if args.dataset == "IoT23":
        datasets = ["val.json",
                    "test_malicious.json", 
                    "test_mixed.json"]
    elif args.dataset == "IoT_traces":
        datasets = ["test_benign.json"]
    elif args.dataset == "IoTID20":
        datasets = ["test_benign.json", "test_mixed.json"]

    for dataset in datasets:
        print(
            f"\n Starting the evaluation for {args.dataset}, {args.graph_type}, ...")
        test(args, dataset)

    test_result_file.close()
