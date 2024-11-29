import utils as u
import torch
import torch.distributed as dist
import numpy as np
import time
import random
import debugpy

# datasets
import anomaly



# taskers
import edge_cls_tasker as ect
import node_cls_tasker as nct
import node_anomaly_tasker as nat

# models
import models as mls
import egcn_h
import egcn_o
import si_vgrnn_model


import splitter as sp
import Cross_Entropy as ce
import Reconstruction_Loss as rl

import trainer as tr

import logger
import wandb
import warnings
import os

warnings.filterwarnings("ignore")


def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower() == 'none':
        if type == 'int':
            return random.randrange(param_min, param_max+1)
        elif type == 'logscale':
            interval = np.logspace(np.log10(param_min),
                                   np.log10(param_max), num=100)
            return np.random.choice(interval, 1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param


def build_random_hyper_params(args):
    if args.model == 'all':
        model_types = ['gcn', 'egcn_o', 'egcn_h',
                       'gruA', 'gruB', 'egcn', 'lstmA', 'lstmB']
        args.model = model_types[args.rank]
    elif args.model == 'all_nogcn':
        model_types = ['egcn_o', 'egcn_h', 'gruA',
                       'gruB', 'egcn', 'lstmA', 'lstmB']
        args.model = model_types[args.rank]
    elif args.model == 'all_noegcn3':
        model_types = ['gcn', 'egcn_h', 'gruA',
                       'gruB', 'egcn', 'lstmA', 'lstmB']
        args.model = model_types[args.rank]
    elif args.model == 'all_nogruA':
        model_types = ['gcn', 'egcn_o', 'egcn_h',
                       'gruB', 'egcn', 'lstmA', 'lstmB']
        args.model = model_types[args.rank]
        args.model = model_types[args.rank]
    elif args.model == 'saveembs':
        model_types = ['gcn', 'gcn', 'skipgcn', 'skipgcn']
        args.model = model_types[args.rank]

    args.learning_rate = random_param_value(
        args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
    # args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')

    if args.model == 'gcn':
        args.num_hist_steps = 0
    else:
        args.num_hist_steps = random_param_value(
            args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')

    args.gcn_parameters['feats_per_node'] = random_param_value(
        args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
    args.gcn_parameters['layer_1_feats'] = random_param_value(
        args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
    # or args.gcn_parameters['layer_2_feats_same_as_l1'].lower() == 'true':
    if args.gcn_parameters['layer_2_feats_same_as_l1']:
        args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
    else:
        args.gcn_parameters['layer_2_feats'] = random_param_value(
            args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
    args.gcn_parameters['lstm_l1_feats'] = random_param_value(
        args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
    # or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower() == 'true':
    if args.gcn_parameters['lstm_l2_feats_same_as_l1']:
        args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
    else:
        args.gcn_parameters['lstm_l2_feats'] = random_param_value(
            args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
    args.gcn_parameters['cls_feats'] = random_param_value(
        args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')
    return args


def build_dataset(args):
    if args.data == 'iot23':
        return anomaly.Anomaly_Dataset(args)
    elif args.data == 'iot_traces':
        return anomaly.Anomaly_Dataset_traces(args)


def build_tasker(args, dataset):
    if args.task == 'edge_cls':
        return ect.Edge_Cls_Tasker(args, dataset)
    elif args.task == 'node_cls':
        return nct.Node_Cls_Tasker(args, dataset)
    elif args.task == 'static_node_cls':
        return nct.Static_Node_Cls_Tasker(args, dataset)
    elif args.task == 'anomaly_detection':
        return nat.Anomaly_Detection_Tasker(args, dataset)
    elif args.task == 'anomaly_detection_iot_traces':
        return nat.Anomaly_Detection_Tasker_IoT_traces(args, dataset)
    else:
        raise NotImplementedError('still need to implement the other tasks')


def build_gcn(args, tasker):
    gcn_args = u.Namespace(args.gcn_parameters)
    gcn_args.feats_per_node = tasker.feats_per_node
    if args.model == 'gcn':
        return mls.Sp_GCN(gcn_args, activation=torch.nn.RReLU()).to(args.device)
    elif args.model == 'skipgcn':
        return mls.Sp_Skip_GCN(gcn_args, activation=torch.nn.RReLU()).to(args.device)
    elif args.model == 'skipfeatsgcn':
        return mls.Sp_Skip_NodeFeats_GCN(gcn_args, activation=torch.nn.RReLU()).to(args.device)
    else:
        assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
        if args.model == 'lstmA':
            return mls.Sp_GCN_LSTM_A(gcn_args, activation=torch.nn.RReLU()).to(args.device)
        elif args.model == 'gruA':
            return mls.Sp_GCN_GRU_A(gcn_args, activation=torch.nn.RReLU()).to(args.device)
        elif args.model == 'lstmB':
            return mls.Sp_GCN_LSTM_B(gcn_args, activation=torch.nn.RReLU()).to(args.device)
        elif args.model == 'gruB':
            return mls.Sp_GCN_GRU_B(gcn_args, activation=torch.nn.RReLU()).to(args.device)
        elif args.model == 'egcn_h':
            return egcn_h.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
        elif args.model == 'skipfeatsegcn_h':
            return egcn_h.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device, skipfeats=True)
        elif args.model == 'egcn_o':
            return egcn_o.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
        elif args.model == 'si_vgrnn':
            return si_vgrnn_model.GCNModelSIGVAE(gcn_args, device=args.device)
        else:
            raise NotImplementedError('need to finish modifying the models')


def build_classifier(args, tasker):
    if not isinstance(tasker, nat.Anomaly_Detection_Tasker) and not isinstance(tasker, nat.Anomaly_Detection_Tasker_IoT_traces):
        if 'node_cls' == args.task or 'static_node_cls' == args.task:
            mult = 1
        else:
            mult = 2
        if 'gru' in args.model or 'lstm' in args.model:
            in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
        elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
            in_feats = (args.gcn_parameters['layer_2_feats'] +
                        args.gcn_parameters['feats_per_node']) * mult
        else:
            in_feats = args.gcn_parameters['layer_2_feats'] * mult

        return mls.Classifier(args, in_features=in_feats, out_features=tasker.num_classes).to(args.device)
    else:
        return mls.Decoder(args)


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
        model_files = [f for f in files if "opt" not in f]
        # Extract numerical indices from the file names
        for model_file in model_files:
            try:
                index = int(model_file.split(".")[0].split("_")[-1])
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

        return os.path.join(directory_path, best_model_file)

    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return None


if __name__ == '__main__':
    parser = u.create_parser()
    args = u.parse_args(parser)

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    global rank, wsize, use_cuda
    args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
    args.device = 'cpu'
    if args.use_cuda:
        args.device = 'cuda'
    else:
        args.device = 'mps'
        #args.device = 'cpu'
    print("use CUDA:", args.use_cuda, "- device:", args.device)
    try:
        dist.init_process_group(backend='mpi')  # , world_size=4
        rank = dist.get_rank()
        wsize = dist.get_world_size()
        print('Hello from process {} (out of {})'.format(
            dist.get_rank(), dist.get_world_size()))
        if args.use_cuda:
            torch.cuda.set_device(rank)  # are we sure of the rank+1????
            print('using the device {}'.format(torch.cuda.current_device()))
    except:
        rank = 0
        wsize = 1
        print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank,
                                                                                   wsize)))

    if args.seed is None and args.seed != 'None':
        seed = 123+rank  # int(time.time())+rank
    else:
        seed = args.seed  # +rank

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed = seed
    args.rank = rank
    args.wsize = wsize

    # Assign the requested random hyper parameters
    args = build_random_hyper_params(args) #Secondo me non si deve eseguire perchè già fatta nel parse_args

    # build the dataset
    dataset = build_dataset(args)

    # build the tasker
    tasker = build_tasker(args, dataset)

    # build the splitter
    splitter = sp.splitter(args, tasker)

    # # build the models
    gcn = build_gcn(args, tasker)
    classifier = build_classifier(args, tasker)
    detector = mls.AnomalyDetector(gcn=gcn,
                                   head=classifier)
    # build a loss
    if isinstance(classifier, mls.Decoder):
        loss = rl.ReconstructionLoss(args=args,
                                     dataset=dataset).to(args.device)
    else:
        loss = ce.Cross_Entropy(args, dataset).to(args.device)

    if args.train:
        # trainer
        trainer = tr.TrainerAnomaly(args,
                                    splitter=splitter,
                                    detector=detector,
                                    comp_loss=loss,
                                    dataset=dataset,
                                    num_classes=tasker.num_classes)
        trainer.train_anomaly()

    if args.test or args.off_line_test:
        if args.off_line_test:
            if args.test_epoch != -1:
                # load best model
                print("Loading detector ....")
                gcnn = torch.load(os.path.join(
                    args.save_folder, args.project_name, f"detector_{args.test_epoch}.pt"))

            else:
                print("Loading best detector ....")
                path = get_best_model(args=args,
                                      directory_path=os.path.join(args.save_folder, args.project_name))

                gcnn = torch.load(path)

            trainer = tr.TrainerAnomaly(args,
                                        splitter=splitter,
                                        detector=detector,
                                        comp_loss=loss,
                                        dataset=dataset,
                                        num_classes=tasker.num_classes)
        print("Test anomaly")
        trainer.test_anomaly(compute_thr=args.compute_threshold)
