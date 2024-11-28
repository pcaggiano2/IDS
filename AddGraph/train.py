# _*_ coding:utf-8 _*_
# @author: Jiajie Lin
# @file: train.py
# @time: 2020/03/13
from os import wait
import os
import csv
import time
import math
import itertools
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from framework.snapshot import *
from framework.model import *
from framework.negative_sample import *
from Graph_dataset import Graph_dataset
from torch.utils.data import DataLoader


def train(args):
    device = args.device

    # Init Models
    Net1 = SpGAT(nfeat=args.hidden, nhid=args.nmid2, nout=args.hidden, dropout=args.dropout, alpha=args.alpha, nheads=args.nb_heads)
    Net2 = HCA(hidden=args.hidden, dropout=args.dropout)
    Net3 = GRU(hidden=args.hidden, dropout=args.dropout)
    Net4 = Score(beta=args.beta, mui=args.mui, hidden=args.hidden, dropout=args.dropout)
    N_S = negative_sample()

    # Set up optimizers for the models
    optimizer1 = optim.Adam(Net1.parameters(), lr=args.lr_)
    optimizer2 = optim.Adam(itertools.chain(Net2.parameters(), Net3.parameters(), Net4.parameters()), lr=args.lr)

    # Learning rate scheduler
    scheduler = StepLR(optimizer1, step_size=10, gamma=0.9)

    # paths definition
    csv_results_folder = os.path.join(args.csv_results_folder, args.graph_type)
    checkpoints_folder = os.path.join(args.checkpoint_folder, args.graph_type)

    print(csv_results_folder)
    print(checkpoints_folder)
    # create csv results folder if it does not exists
    if not os.path.exists(csv_results_folder):
        os.makedirs(csv_results_folder)

    # create checkpoint folder if it does not exists
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    # write training and validation results in a csv file
    train_result_file = open(os.path.join(
        csv_results_folder, 'train_result.csv'), "w", newline='')
    train_writer = csv.writer(train_result_file)
    val_result_file = open(os.path.join(
        csv_results_folder, 'validation_result.csv'), "w", newline='')
    val_writer = csv.writer(val_result_file)
    train_writer.writerow(
        ['epoch', 'loss', 'struct_loss', 'feat_loss', 'time_per_epoch'])
    val_writer.writerow(['epoch', 'loss', 'struct_loss', 'feat_loss'])

    # Dataloaders definition
    json_training_set = os.path.join(args.json_folder, "train.json")
    json_validation_set = os.path.join(args.json_folder, "val.json")

    # dataset_path, json_path, representation
    train = Graph_dataset(args.dataset_folder,
                          json_training_set,
                          args.graph_type)
    # Passare min_max perchè non è in base
    val = Graph_dataset(args.dataset_folder,
                        json_validation_set, 
                        args.graph_type)

    train_dataloader = DataLoader(
        train,
        batch_size=1,  # args.batch_size,
        num_workers=0,
        shuffle=True)

    val_dataloader = DataLoader(
        val,
        batch_size=1,  # args.batch_size,
        num_workers=0,
        shuffle=True)

    # Load snapshots
    l_train = int(train.len())
    nodes = int(train.n_nodes)

    l_val = int(val.len())

    # Start training
    best_val_loss = float("inf")
    best_epoch = 0
    consecutive_bad_epochs = 0
    step_sample = 0    

    """Training loop for the models."""
    t = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()
        print("\n ------- Epoch ", epoch, " - at: ", start_time)
        loss_batch = 0.0
        Net1.train(True)
        Net2.train(True)
        Net3.train(True)
        Net4.train(True)
        N_S.train(True)

        H_list = torch.zeros(1, nodes, args.hidden)
        H_ = torch.zeros((args.w, nodes, args.hidden))

        # Initialize hidden states
        stdv = 1. / math.sqrt(args.hidden)
        H_list[-1][:l_train, :].data.uniform_(-stdv, stdv)

        U_adj = torch.zeros((nodes, nodes), dtype=torch.int16)
        loss_a = torch.zeros(1)

        for snapshot, labels in train_dataloader:
            step_sample += 1

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            snapshot = torch.tensor(snapshot)
            H = H_list[-1]
            H_ = torch.zeros((args.w, nodes, args.hidden))

            # Update adjacency matrix and model inputs
            U_adj, adj, Adj = update_adj(U_adj=U_adj, snapshot=snapshot, nodes=nodes)
            
            H = H.to(device, dtype=torch.float32)
            Adj = Adj.to(device)
            snapshot = snapshot.to(device)
            wait(100000)
            # Forward pass through the models
            current = Net1(x=H, Adj=Adj, adj=adj)
            short = Net2(C=H_)
            Hn = Net3(current=current, short=short)
            H_list = torch.cat([H_list, Hn.unsqueeze(0)], dim=0)

            # Compute loss
            n_loss = N_S(adj=adj, U_adj=U_adj, snapshot=snapshot, H=Hn, f=Net4, arg=(args.device=="cuda"))
            loss1 = args.weight_decay * (Net1.loss() + Net2.loss() + Net3.loss() + Net4.loss())
            loss2 = torch.sum(torch.where((args.gama + n_loss) >= 0, (args.gama + n_loss), torch.zeros(1)))

            loss = loss1 + loss2

            loss_a[0] += loss 

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            #print(f'Epoch: {epoch}, Step: {i}, Loss: {loss.item()}')
        
        loss_a.mean()
        scheduler.step()
        print(f'Average loss for epoch {epoch}: {loss_a.item()}')
        print(f'Time taken: {time.time() - t:.2f}s')
        Net1.train(False)
        Net2.train(False)
        Net3.train(False)
        Net4.train(False)
        N_S.train(False)

        print("\n Computing training results...")
        total_train_loss = 0.0
        with torch.no_grad():
            for snapshot, labels in train_dataloader:
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                snapshot = torch.tensor(snapshot)
                H = H_list[-1]
                H_ = torch.zeros((args.w, nodes, args.hidden))

                # Update adjacency matrix and model inputs
                U_adj, adj, Adj = update_adj(U_adj=U_adj, snapshot=snapshot, nodes=nodes)

                H = H.to(device, dtype=torch.float32)
                Adj = Adj.to(device)
                snapshot = snapshot.to(device)

                # Forward pass through the models
                current = Net1(x=H, Adj=Adj, adj=adj)
                short = Net2(C=H_)
                Hn = Net3(current=current, short=short)
                H_list = torch.cat([H_list, Hn.unsqueeze(0)], dim=0)

                # Compute loss
                n_loss = N_S(adj=adj, U_adj=U_adj, snapshot=snapshot, H=Hn, f=Net4, arg=(args.device=="cuda"))
                loss1 = args.weight_decay * (Net1.loss() + Net2.loss() + Net3.loss() + Net4.loss())
                loss2 = torch.sum(torch.where((args.gama + n_loss) >= 0, (args.gama + n_loss), torch.zeros(1)))

                total_train_loss = loss1 + loss2

            average_train_loss = total_train_loss / len(train_dataloader)
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            train_writer.writerow(
                [epoch, average_train_loss, elapsed_time])

            print("\nTrain Loss:", average_train_loss,
                  " - elapsed time: ", elapsed_time)

            # Validation
            print("\n Validating...")
            total_val_loss = 0.0
            Net1.eval()
            Net2.eval()
            Net3.eval()
            Net4.eval()
            N_S.eval()

            with torch.no_grad():
                for snapshot, labels in val_dataloader:
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    snapshot = torch.tensor(snapshot)
                    H = H_list[-1]
                    H_ = torch.zeros((args.w, nodes, args.hidden))

                    # Update adjacency matrix and model inputs
                    U_adj, adj, Adj = update_adj(U_adj=U_adj, snapshot=snapshot, nodes=nodes)
                    
                    H = H.to(device, dtype=torch.float32)
                    Adj = Adj.to(device)
                    snapshot = snapshot.to(device)

                    # Forward pass through the models
                    current = Net1(x=H, Adj=Adj, adj=adj)
                    short = Net2(C=H_)
                    Hn = Net3(current=current, short=short)
                    H_list = torch.cat([H_list, Hn.unsqueeze(0)], dim=0)

                    # Compute loss
                    n_loss = N_S(adj=adj, U_adj=U_adj, snapshot=snapshot, H=Hn, f=Net4, arg=(args.device=="cuda"))
                    loss1 = args.weight_decay * (Net1.loss() + Net2.loss() + Net3.loss() + Net4.loss())
                    loss2 = torch.sum(torch.where((args.gama + n_loss) >= 0, (args.gama + n_loss), torch.zeros(1)))

                    total_val_loss = loss1 + loss2

            average_val_loss = total_val_loss / len(val_dataloader)

            val_writer.writerow(
                [epoch, average_val_loss])

            print("\nVal Loss:", average_val_loss)

            if average_val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = average_val_loss
                #SAVE BEST MODEL
                torch.save({'Net1': Net1.state_dict(),
                            'Net2': Net2.state_dict(),
                            'Net3': Net3.state_dict(),
                            'Net4': Net4.state_dict(),
                            'H_list': H_list,
                            'loss_a': loss_a,
                            'U_adj': U_adj
                        }, checkpoints_folder +
                            "/AddGraph_model_{}".format(best_epoch))
                consecutive_bad_epochs = 0
            else:
                consecutive_bad_epochs += 1

            if consecutive_bad_epochs == args.patience:
                print(
                    f'Validation loss has not improved for {args.patience} epochs. Early stopping...')
                break

    val_result_file.close()
    train_result_file.close()

if __name__ == "__main__":


    # Set up argument parser for training settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--lr_', type=float, default=0.0008, help='Learning rate for first model.')
    parser.add_argument('--weight_decay', type=float, default=5e-7, help='L2 weight decay on parameters.')
    parser.add_argument('--hidden', type=int, default=50, help='Number of hidden units.')
    parser.add_argument('--nmid2', type=int, default=70, help='Number of units for the second mid-layer.')
    parser.add_argument('--beta', type=float, default=3.0, help='Hyper-parameter for the score function.')
    parser.add_argument('--mui', type=float, default=0.5, help='Hyper-parameter for the score function.')
    parser.add_argument('--gama', type=float, default=0.6, help='Hyper-parameter for the score function.')
    parser.add_argument('--w', type=int, default=3, help='Hyper-parameter for the score function.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky ReLU activation.')
    parser.add_argument('--nb_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Number of graphs per batch')
    parser.add_argument('--device',
                        type=str,
                        #default='cuda',
                        default='mps',
                        help='GPU = cuda/CPU = cpu')
    parser.add_argument('--patience',
                        type=int,
                        default=20,
                        help='Number of epoch for early stopping')

    parser.add_argument("--dataset_folder",
                        type=str,
                        help="Dataset folder from which take the graphs")

    parser.add_argument("--json_folder",
                        type=str,
                        help="Dataset folder in json format from which take the dataset split")

    parser.add_argument("--graph_type",
                        type=str,
                        help="Graph type to consider (similarity_graph/tdg_graph)")

    parser.add_argument("--checkpoint_folder",
                        type=str,
                        help="Folder where to save checkpoints")

    parser.add_argument("--csv_results_folder",
                        type=str,
                        help="Folder where to save results as csv file")

    parser.add_argument("--model",
                        type=str)
    
    parser.add_argument("--dataset",
                        type=str,
                        default="IoT23")

    args = parser.parse_args()

    # Set device for training
    if args.device == "cuda":
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        print('Using CUDA.')
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

    train(args)

    #     # Dataloaders definition
    # json_training_set = os.path.join(args.json_folder, "train.json")
    # json_validation_set = os.path.join(args.json_folder, "val.json")

    # # dataset_path, json_path, representation
    # train = Graph_dataset(args.dataset_folder,
    #                       json_training_set,
    #                       args.graph_type)
    # # Passare min_max perchè non è in base
    # val = Graph_dataset(args.dataset_folder,
    #                     json_validation_set, 
    #                     args.graph_type)
    

    # print(train.get(0))

    # train_dataloader = DataLoader(
    #     train,
    #     batch_size=1,  # args.batch_size,
    #     num_workers=0,
    #     shuffle=True)

    # val_dataloader = DataLoader(
    #     val,
    #     batch_size=1,  # args.batch_size,
    #     num_workers=0,
    #     shuffle=True)

