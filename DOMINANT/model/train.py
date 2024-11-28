import torch
import os
import csv
import argparse
import time
from DOMINANT import DOMINANT
from functional import objective_function
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader
from Graph_dataset import Graph_dataset
import warnings
warnings.filterwarnings("ignore")


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

    score, train_attr_error, train_struct_error = objective_function(x,
                                                                     x_,
                                                                     s,
                                                                     s_,
                                                                     alpha)

    return score, train_attr_error.detach().cpu(), train_struct_error.detach().cpu()


def collate_fn(data):
    print(data)


def train(args):
    # paths definition
    csv_results_folder = os.path.join(args.csv_results_folder, args.graph_type)
    checkpoints_folder = os.path.join(args.checkpoint_folder, args.graph_type)

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
                          args.graph_type,
                          args.normalize, 
                          args.min_max)
    # Passare min_max perchè non è in base
    val = Graph_dataset(args.dataset_folder,
                        json_validation_set, 
                        args.graph_type,
                        args.normalize, 
                        args.min_max)

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

    device = None
    # check if GPU is available
    if args.device == 'cuda':
        device = torch.device(args.device)
        print("Using GPU:", device)
    else:
        device = torch.device("mps")
        print("Using GPU:", device)
    

    if args.graph_type == "tdg_graph" or args.graph_type == "sim_graph":
        in_dim = 57

    # model definition and loading to GPU
    model = DOMINANT(in_dim=in_dim,
                     hid_dim=args.hidden_dim,
                     encoder_layers=args.encoder_layers,
                     decoder_layers=args.decoder_layers,
                     dropout=args.dropout).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # start training
    best_val_loss = float("inf")
    best_epoch = 0
    consecutive_bad_epochs = 0
    to_log = dict()
    step_sample = 0
    step_batch = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        print("\n ------- Epoch ", epoch, " - at: ", start_time)
        model.train(True)
        loss_batch = 0.0
        for sample in train_dataloader:
            step_sample += 1
            # Obtain the dense adjacency matrix of the graph
            process_graph(sample)
            # Forward pass
            x = sample.x.to(device)
            s = sample.s.to(device)
            edge_index = sample.edge_index.to(device)

            x_, s_ = model(x, edge_index)

            # Calculate loss
            score, _, _ = compute_node_anomaly_score(x, x_, s, s_, args.alpha)
            loss_batch += torch.mean(score)
            # Backpropagation
            if step_sample % args.batch_size == 0:
                step_batch += 1
                optimizer.zero_grad()
                loss_batch = loss_batch/args.batch_size
                loss_batch.backward()
                optimizer.step()
                loss_batch = 0.0

        model.train(False)

        print("\n Computing training results...")
        total_train_loss = 0.0
        total_train_attr_error = 0.0
        total_train_struct_error = 0.0
        with torch.no_grad():
            for batch in train_dataloader:
                process_graph(batch)
                x = batch.x.to(device)
                s = batch.s.to(device)
                edge_index = batch.edge_index.to(device)

                x_, s_ = model(x, edge_index)

                train_loss, train_attr_error, train_struct_error = compute_node_anomaly_score(
                    x, x_, s, s_, args.alpha)

                total_train_loss += torch.mean(
                    train_loss.detach().cpu()).item()
                total_train_attr_error += torch.mean(train_attr_error).item()
                total_train_struct_error += torch.mean(
                    train_struct_error).item()

            average_train_loss = total_train_loss / len(train_dataloader)
            average_attr_error = total_train_attr_error / len(train_dataloader)
            average_struct_error = total_train_struct_error / \
                len(train_dataloader)

            end_time = time.time()
            elapsed_time = end_time - start_time

            train_writer.writerow(
                [epoch, average_train_loss, average_attr_error, average_struct_error, elapsed_time])

            print("\nTrain Loss:", average_train_loss,
                  " - Train Struct Loss:", average_attr_error,
                  " - Train Feat Loss:", average_struct_error,
                  " - elapsed time: ", elapsed_time)

            # Validation
            print("\n Validating...")
            total_val_loss = 0.0
            total_val_attr_error = 0.0
            total_val_struct_error = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    process_graph(batch)
                    x = batch.x.to(device)
                    s = batch.s.to(device)
                    edge_index = batch.edge_index.to(device)

                    x_, s_ = model(x, edge_index)

                    # Calculate loss
                    val_loss, val_attr_error, val_struct_error = compute_node_anomaly_score(
                        x, x_, s, s_, args.alpha)
                    total_val_loss += torch.mean(
                        val_loss.detach().cpu()).item()
                    total_val_attr_error += torch.mean(val_attr_error).item()
                    total_val_struct_error += torch.mean(
                        val_struct_error).item()

                average_val_loss = total_val_loss / len(val_dataloader)
                average_val_attr_error = total_val_attr_error / \
                    len(val_dataloader)
                average_val_struct_erro = total_val_struct_error / \
                    len(val_dataloader)

                val_writer.writerow(
                    [epoch, average_val_loss, average_val_attr_error, average_val_struct_erro])

                print("\nVal Loss:", average_val_loss,
                      " - Val Struct Loss:", average_val_attr_error,
                      " - Val Feat Loss:", average_val_struct_erro)

                print("\nVal Loss:", average_val_loss)

                if average_val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = average_val_loss
                    torch.save(model.state_dict(), checkpoints_folder +
                               "/DOMINANT_model_{}".format(best_epoch))
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        help='Training epochs')

    parser.add_argument('--patience',
                        type=int,
                        default=20,
                        help='Number of epoch for early stopping')

    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Number of graphs per epoch (must be 1 for trajectory graph representation)')

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
                        default=1e-4,
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

    parser.add_argument("--dataset_folder",
                        type=str,
                        help="Dataset folder from which take the graphs")

    parser.add_argument("--json_folder",
                        type=str,
                        help="Dataset folder in json format from which take the dataset split")

    parser.add_argument("--graph_type",
                        type=str,
                        help="Graph type to consider (similarity_graph/trajectory_graph/etdg_graph)")

    parser.add_argument("--checkpoint_folder",
                        type=str,
                        help="Folder where to save checkpoints")

    parser.add_argument("--csv_results_folder",
                        type=str,
                        help="Folder where to save results as csv file")

    parser.add_argument("--model",
                        type=str,
                        default=None)
    
    parser.add_argument("--dataset",
                        type=str,
                        default="IoT23")
    
    parser.add_argument("--min_max",
                        type=str,
                        help="Path to the min and max numpy arrays")
    
    parser.add_argument("--normalize",
                        type=int,
                        help="Normalize the dataset")

    parser.add_argument("--debug",
                        action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger")
        debugpy.wait_for_client()

    train(args)

    # # Dataloaders definition
    # json_training_set = os.path.join(args.json_folder, "train.json")
    # json_validation_set = os.path.join(args.json_folder, "val.json")

    # # dataset_path, json_path, representation
    # train = Graph_dataset(args.dataset_folder,
    #                       json_training_set,
    #                       args.graph_type,
    #                       args.normalize,
    #                       args.min_max)
    # # Passare min_max perchè non è in base
    # val = Graph_dataset(args.dataset_folder,
    #                     json_validation_set, 
    #                     args.graph_type,
    #                     args.normalize,
    #                     args.min_max)

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


    # print(train.get(0))
    # print(val.get(0))
