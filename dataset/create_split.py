import os
import pandas as pd
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from math import floor
import json
import functools
import multiprocessing as mp
from tqdm import tqdm
import multiprocessing.pool
from collections import OrderedDict
import pickle
import networkx as nx
import csv


DATASET = OrderedDict()
DATASET_TRAIN = OrderedDict()  # contains sequence of full_benign graphs
DATASET_VAL = OrderedDict()  # contains sequence of full_benign graphs
DATASET_TEST_MIXED = OrderedDict()
DATASET_TEST_FULL_MAL = OrderedDict()
DATASET_TEST_BENIGN = OrderedDict()


def novel_sequence(capture_name, previous_indx, graph_number, representation_name, graph_type):
    # Check if the difference between previous_indx e graph_number is due to the absence of graphs or no
    other_type_graph = False
    for graph_t in DATASET[capture_name][representation_name].keys():
        if graph_type != graph_t:
            graph_list = DATASET[capture_name][representation_name][graph_t]
            if len(graph_list) != 0:
                # path = os.path.join(BASE_PATH, capture_name,
                #                     representation_name, graph_t)
                # check if the absent graph is in other representations
                for indx in range(graph_number-previous_indx):
                    if f"graph_{previous_indx+indx}.pkl" in graph_list:
                        return True


def filter_list(capture_name, benign_list, mixed_list, representation):
    first_benign = int(benign_list[0].split('_')[-1].split('.')[0])
    second_benign = int(benign_list[1].split('_')[-1].split('.')[0])
    first_mixed = int(mixed_list[0].split('_')[-1].split('.')[0])
    second_mixed = int(mixed_list[1].split('_')[-1].split('.')[0])
    third_mixed = int(mixed_list[2].split('_')[-1].split('.')[0])

    new_mixed_list = mixed_list
    # If the two consecutive benign do not belong to the same sequence
    if novel_sequence(capture_name, first_benign, second_benign, representation, "full_benign"):
        if first_mixed < second_benign and first_mixed > first_benign:
            # check if the lenght mixed sequence is less than two
            if novel_sequence(capture_name, first_mixed, second_mixed, representation, "mixed"):
                # remove the first mixed
                new_mixed_list.pop(0)
            # check if the second and third belong to the same sequence
            elif novel_sequence(capture_name, second_mixed, third_mixed, representation, "mixed"):
                # remove the first and second mixed
                new_mixed_list.pop(0)
                new_mixed_list.pop(0)
    else:
        # check if the lenght mixed sequence is less than two
        if novel_sequence(capture_name, first_mixed, second_mixed, representation, "mixed"):
            # remove the first mixed
            new_mixed_list.pop(0)
        # check if the second and third belong to the same sequence
        elif novel_sequence(capture_name, second_mixed, third_mixed, representation, "mixed"):
            # remove the first and second mixed
            new_mixed_list.pop(0)
            new_mixed_list.pop(0)

    return benign_list, new_mixed_list


def compute_sequences(capture_name, graph_list, representation, graph_type):
    previous_indx = -1
    start = -1
    end = -1
    start_with_malware = False
    current_sequence = []
    sequences = []
    for graph_path in graph_list:
        # print(graph_name)
        graph_name = graph_path.split('/')[-1]
        # take graph_number
        graph_number = int(
            graph_name.split('_')[-1].split('.')[0])
        if graph_number == previous_indx+1 or previous_indx == -1:
            if previous_indx == -1 and graph_number != 0:
                start = graph_number
                if novel_sequence(capture_name, 0, graph_number, representation, graph_type):
                    start_with_malware = True

            current_sequence.append(graph_path)
            previous_indx = graph_number
        else:
            if novel_sequence(capture_name, previous_indx, graph_number, representation, graph_type):
                end = previous_indx
                sequences.append(current_sequence)
                current_sequence = []
                # start novel sequence
                current_sequence.append(graph_path)
                previous_indx = graph_number
                start = graph_number
            else:
                current_sequence.append(graph_path)
    sequences.append(current_sequence)

    return sequences, start_with_malware


def split_train_test_val(dataset_dict: dict = None, representation_list: list = ["tdg_graph"], train_val_split: float = 0.80, temporal_split: float = 0.80, temporal: bool = False, dataset_path: str = None, only_test: bool = False, out_path=None):
    # for each capture
    for representation in representation_list:
        num_train_graphs = 0.0
        num_val_graphs = 0.0
        num_test_mixed = 0.0
        num_test_mal = 0.0
        num_test_benign = 0.0
        for capture_name in dataset_dict.keys():
            if not only_test:
                # if the capture is benign split between train and val based on temporal split
                if "Honeypot" in capture_name:
                    if DATASET_TRAIN.get(capture_name) is None:
                        DATASET_TRAIN[capture_name] = []
                        DATASET_VAL[capture_name] = []

                    graphs_path = dataset_dict[capture_name][representation]["full_benign"]
                    train_number = floor(len(graphs_path)*temporal_split)
                    graphs_train = graphs_path[:train_number]
                    graphs_val = graphs_path[train_number:]
                    DATASET_TRAIN[capture_name].append(
                        graphs_train)
                    DATASET_VAL[capture_name].append(graphs_val)
                    num_train_graphs += len(graphs_train)
                    num_val_graphs += len(graphs_val)

                else:

                    full_benign_graphs = dataset_dict[capture_name][representation]["full_benign"]
                    full_mixed_graphs = dataset_dict[capture_name][representation]["mixed"]
                    full_malicious_graphs = dataset_dict[capture_name][representation]["full_malicious"]

                    print(f"Total full_benign_graphs: {len(full_benign_graphs)}")

                    if len(full_benign_graphs) > 2 and len(full_mixed_graphs) > 3:
                        print("Perform filtering")
                        full_benign_graphs, full_mixed_graphs = filter_list(
                            capture_name=capture_name, benign_list=full_benign_graphs, mixed_list=full_mixed_graphs, representation=representation)
                        print(f"Total full_benign_graphs after filtering: {len(full_benign_graphs)}")


                    if len(full_malicious_graphs) != 0:
                        DATASET_TEST_FULL_MAL[capture_name] = []
                        num_test_mal += len(full_malicious_graphs)
                        sequences, _ = compute_sequences(capture_name=capture_name,
                                                         graph_list=full_malicious_graphs,
                                                         representation=representation,
                                                         graph_type="full_malicious")
                        for sequence in sequences:
                            DATASET_TEST_FULL_MAL[capture_name].append(
                                sequence)

                    if len(full_mixed_graphs) != 0:
                        DATASET_TEST_MIXED[capture_name] = []
                        num_test_mixed += len(full_mixed_graphs)
                        sequences, _ = compute_sequences(
                            capture_name=capture_name,
                            graph_list=full_mixed_graphs,
                            representation=representation,
                            graph_type="mixed")
                        for sequence in sequences:
                            DATASET_TEST_MIXED[capture_name].append(
                                sequence)

                    if len(full_benign_graphs) != 0:

                        ### --- FULL BENIGN --- ###
                        # create a sequence of full benign graphs that we have at the beginning
                        #print(full_benign_graphs)

                        sequences, start_with_malware = compute_sequences(
                            capture_name=capture_name,
                            graph_list=full_benign_graphs,
                            representation=representation,
                            graph_type="full_benign")
                        
                        print(f"Start with malware: {start_with_malware}")

                        if not start_with_malware:
                            # the first sequence can go in training
                            first_sequence = sequences[0]
                            DATASET_TRAIN[capture_name] = []
                            if len(first_sequence) > 1:
                                DATASET_VAL[capture_name] = []

                            if len(first_sequence) == 1:
                                DATASET_TRAIN[capture_name].append(
                                    first_sequence)
                                num_train_graphs += 1

                            elif len(first_sequence) == 2:
                                DATASET_TRAIN[capture_name].append(
                                    [first_sequence[0]])
                                DATASET_VAL[capture_name].append(
                                    [first_sequence[1]])
                                num_train_graphs += 1
                                num_val_graphs += 1
                            else:
                                train_number = floor(
                                    len(first_sequence)*temporal_split)
                                graphs_train = first_sequence[:train_number]
                                graphs_val = first_sequence[train_number:]
                                DATASET_TRAIN[capture_name].append(
                                    graphs_train)
                                DATASET_VAL[capture_name].append(graphs_val)
                                num_train_graphs += len(graphs_train)
                                num_val_graphs += len(graphs_val)
                            # all the other sequences go to test benign
                            for indx, sequence in enumerate(sequences[1:]):
                                if indx == 0:
                                    DATASET_TEST_BENIGN[capture_name] = []
                                DATASET_TEST_BENIGN[capture_name].append(
                                    sequence)
                                num_test_benign += len(sequence)
                        else:
                            for indx, sequence in enumerate(sequences):
                                if indx == 0:
                                    DATASET_TEST_BENIGN[capture_name] = []
                                DATASET_TEST_BENIGN[capture_name].append(
                                    sequence)
                                num_test_benign += len(sequence)
            else:
                full_benign_graphs = dataset_dict[capture_name][representation]["full_benign"]
                full_mixed_graphs = dataset_dict[capture_name][representation]["mixed"]
                full_malicious_graphs = dataset_dict[capture_name][representation]["full_malicious"]
                if len(full_malicious_graphs) != 0:
                    DATASET_TEST_FULL_MAL[capture_name] = []
                    num_test_mal += len(full_malicious_graphs)
                    sequences, start_with_malware = compute_sequences(
                        capture_name=capture_name,
                        graph_list=full_malicious_graphs,
                        representation=representation,
                        graph_type="full_malicious")
                    for sequence in sequences:
                        DATASET_TEST_FULL_MAL[capture_name].append(sequence)
                if len(full_mixed_graphs) != 0:
                    DATASET_TEST_MIXED[capture_name] = []
                    num_test_mixed += len(full_mixed_graphs)
                    sequences, start_with_malware = compute_sequences(
                        capture_name=capture_name,
                        graph_list=full_mixed_graphs,
                        representation=representation,
                        graph_type="mixed")
                    for sequence in sequences:
                        DATASET_TEST_MIXED[capture_name].append(sequence)
                if len(full_benign_graphs) != 0:
                    DATASET_TEST_BENIGN[capture_name] = []
                    num_test_benign += len(full_benign_graphs)
                    sequences, start_with_malware = compute_sequences(
                        capture_name=capture_name,
                        graph_list=full_benign_graphs,
                        representation=representation,
                        graph_type="full_malicious")
                    for sequence in sequences:
                        DATASET_TEST_BENIGN[capture_name].append(sequence)

        print(f"Train {num_train_graphs} - Val {num_val_graphs} - Test benign {num_test_benign} - Test malign {num_test_mal} - Test mixed {num_test_mixed}")

        typ_val = representation.split("_")[0]
   
        # Write IOT23 Dataset
        if "IoT23" in dataset_path:
            save_folder = "IoT23_dataset_split_"+typ_val if not temporal else "IoT23_dataset_split_temporal_"+typ_val
            save_folder = os.path.join(out_path, save_folder)
            os.makedirs(save_folder, exist_ok=True)
        elif "IoT_traces" in dataset_path:
            save_folder = "IoT_traces_dataset_split_"+typ_val if not temporal else "IoT_traces_dataset_split_temporal_"+typ_val
            save_folder = os.path.join(out_path, save_folder)
            os.makedirs(save_folder, exist_ok=True)
        elif "IoTID20" in dataset_path:
            save_folder = "IoTID20_dataset_split_"+typ_val if not temporal else "IoTID20_dataset_split_temporal_"+typ_val
            save_folder = os.path.join(out_path, save_folder)
            os.makedirs(save_folder, exist_ok=True)
        elif "Bot-IoT" in dataset_path:
            save_folder = "Bot-IoT_dataset_split_"+typ_val if not temporal else "Bot-IoT_dataset_split_temporal_"+typ_val
            save_folder = os.path.join(out_path, save_folder)
            os.makedirs(save_folder, exist_ok=True)

        with open(f"{save_folder}/train.json", "w") as outfile:
            # json_data refers to the above JSON
            json.dump(DATASET_TRAIN, outfile)
        with open(f"{save_folder}/val.json", "w") as outfile:
            # json_data refers to the above JSON
            json.dump(DATASET_VAL, outfile)
        with open(f"{save_folder}/test_benign.json", "w") as outfile:
            # json_data refers to the above JSON
            json.dump(DATASET_TEST_BENIGN, outfile)
        with open(f"{save_folder}/test_malicious.json", "w") as outfile:
            # json_data refers to the above JSON
            json.dump(DATASET_TEST_FULL_MAL, outfile)
        with open(f"{save_folder}/test_mixed.json", "w") as outfile:
            # json_data refers to the above JSON
            json.dump(DATASET_TEST_MIXED, outfile)


if __name__ == "__main__":
    import debugpy
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--data_path', type=str,
                        help="Path to folder containing graph pickle files")

    parser.add_argument('--out_path', type=str,
                        help="Path to folder to save the output split json files")
    parser.add_argument('--train_val_split', type=float,
                        default=0.80)
    parser.add_argument('--temporal_split', type=float,
                        default=0.80,)
    parser.add_argument('--only_test', action='store_true')

    args = parser.parse_args()
    data_path = args.data_path
    out_path = args.out_path

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    print("DATA PATH: " + data_path)
    # create capture list
    if "IoT23" in data_path:
        #result_paths = glob.glob(f"{data_path}/CTU-*")
        result_paths = glob.glob(f"{data_path}/*") #NEW CODE
    elif "IoT_traces" in data_path:
        result_paths = glob.glob(f"{data_path}/*")
    elif "IoTID20" in data_path:
        result_paths = glob.glob(f"{data_path}/*")
    elif "Bot-IoT" in data_path:
        result_paths = glob.glob(f"{data_path}/*")
  

    #files_to_remove = [f"{data_path}/CTU-IoT-Malware-Capture-49-1"]
    capture_paths = result_paths
    print(capture_paths)
    # COMMENTED CODE BECAUSE I'M USIGN CLEAN IOT23 Dataset
    # if "IoT23" in data_path:
    #     capture_paths = sorted(capture_paths, key=lambda x: int(
    #         x.split('-')[-2]))

    representatios = []
    for capt_indx, capture_path in enumerate(capture_paths):
        # if indx < 1:
        # print(capt_indx)
        capture_name = capture_path.split('/')[-1]
        print(f"Considering capture {capture_name}")

        if DATASET.get(capture_name) is None:
            DATASET[capture_name] = OrderedDict()
        #if 'base' in data_path:
        #    representatios = ['etdg_graph', 'tdg_graph']
        #else:
        #    representatios = ['etdg_graph']
        #Solo per 10k representatios = ['tdg_graph']
        representatios = ['tdg_graph','sim_graph']
        # for each representation
        for representation_name in representatios:
            print(f"Considering representation {representation_name}")

            if DATASET[capture_name].get(representation_name) is None:
                DATASET[capture_name][representation_name] = OrderedDict()

            # for each graph type:
            for graph_type in ['full_benign', 'full_malicious', 'mixed']:
                # print(f"Considering graph type {graph_type}")

                if DATASET[capture_name][representation_name].get(graph_type) is None:
                    DATASET[capture_name][representation_name][graph_type] = []

                # file list in current folder
                if args.temporal:
                    graphs_path = glob.glob(
                        f"{capture_path}/{representation_name}/{graph_type}/full_graph_*.pkl")
                else:
                    graphs_path = glob.glob(
                        f"{capture_path}/{representation_name}/{graph_type}/graph_*.pkl")
                # order graphs_path
                graphs_path = sorted(
                    graphs_path, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                # for each graph
                previous_indx = -1

                for indx, graph_path in enumerate(graphs_path):

                    DATASET[capture_name][representation_name][graph_type].append(
                        graph_path.split('/')[-1])
    print(args.only_test)
    split_train_test_val(dataset_dict=DATASET,
                         representation_list=representatios,
                         train_val_split=args.train_val_split,
                         temporal_split=args.temporal_split,
                         temporal=args.temporal,
                         dataset_path=args.data_path,
                         only_test=args.only_test,
                         out_path=out_path)
