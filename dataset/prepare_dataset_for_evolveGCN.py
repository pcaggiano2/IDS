import os
import pickle
import numpy as np
import json
from tqdm import tqdm
from collections import OrderedDict
from multiprocessing import Pool
import functools
import re

IP_PORT_MAT_INDX_MAP = dict()


def open_graph_pkl(file_path, capture_name):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def add_full_adj(result_path, representation, graph_type, capture_name, graph_path):
    # check if the full graph already exist
    save_folder = f"{result_path}/{representation}_graph/{graph_type}"
    graph_number = int(graph_path.split(
        '/')[-1].split('.')[0].split('_')[-1])
    graph_name = f"graph_{graph_number}.pkl"
    save_file_path = os.path.join(
        result_path,
        f"{representation}_graph",
        graph_type,
        graph_name
    )
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    if True:  # data.get("full_adj", None) is None:
        # number of distinct nodes
        n_nodes = len(
            list(IP_PORT_MAT_INDX_MAP[capture_name].keys()))
        full_adj = []
        full_adj_labels = []
        full_nodes_features = []
        full_nodes_labels = []
        for old_indx in data['node_id_to_port_ip'].keys():
            # get the novel indx
            novel_indx = IP_PORT_MAT_INDX_MAP[capture_name][str(
                data['node_id_to_port_ip'][old_indx])]
            # update node feature
            full_nodes_features.append([old_indx, novel_indx])
            full_nodes_labels.append(
                [novel_indx, data['node_labels'][old_indx]])

            # update adjacence matrix
            old_adj_row = data['adj'][old_indx]
            neighbors_old_indx = np.where(old_adj_row == 1)[0]
            for neighbor_old_indx in neighbors_old_indx:
                # neighbor_new_indx = IP_PORT_MAT_INDX_MAP[capture_name][
                #     str(data['node_id_to_port_ip'][neighbor_old_indx])]
                # full_adj[novel_indx][neighbor_new_indx] = data['adj'][old_indx][neighbor_old_indx]
                # there is an edge between old_indx and new_indx
                if data['adj'][old_indx][neighbor_old_indx] != 0:
                    neighbor_new_indx = IP_PORT_MAT_INDX_MAP[capture_name][
                        str(data['node_id_to_port_ip'][neighbor_old_indx])]
                    full_adj.append(
                        [novel_indx, neighbor_new_indx])
                    full_adj_labels.append(data['flow_label_matrix'][old_indx][neighbor_old_indx])

        # create a complete graph
        # full_graph = {"node_features": full_nodes_features,
        #               "adj": full_adj,
        #               "node_labels": full_nodes_labels,
        #               "n_nodes": n_nodes}
        data["n_nodes"] = n_nodes
        data["full_adj"] = full_adj
        data["full_adj_labels"] = full_adj_labels
        data["features_new_old_map"] = full_nodes_features
        data["label_new_indx_label"] = full_nodes_labels
        with open(save_file_path, "wb") as handle:
            print(f"Dump full graph {save_file_path}")
            pickle.dump(data, handle, protocol=4)
    else:
        print("skip")


if __name__ == "__main__":
    import debugpy
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data_path', type=str,
                        default='/user/apaolillo/Output_Grafi/150000/IoT23/base')
                        #default='D:/Alessandro/Universita/Salerno/Magistrale/Tesi/Output_Grafi/60000/IoTID20/graph_no_approx/')
                        
    #parser.add_argument('--graph_repre', type=str,default="eTDG", help="Which kind of graphs to create [TDG, eTDG, TRAJ, SIM, ALL]")
    args = parser.parse_args()

    # Open Trajectory Graphs for checking their content
    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    data_path = args.data_path
    #graph_type = args.graph_repre

    #
    if "IoT23" in data_path:
        result_paths = glob.glob(f"{data_path}/*")
    elif "IoT_traces" in data_path:
        result_paths = glob.glob(f"{data_path}/*")  
    elif "IoTID20" in data_path:
        result_paths = glob.glob(f"{data_path}/*")
    elif "Bot-IoT" in data_path:
        result_paths = glob.glob(f"{data_path}/*")

    dimensions = []
    for capture_path in result_paths:
        size = 0
        for _, dirs_graph_repre, _ in os.walk(capture_path):
            for dir_graph_repre in dirs_graph_repre:
                for _, dirs_graph_type, _ in os.walk(os.path.join(capture_path, dir_graph_repre)):
                    for graph_type in dirs_graph_type:
                        for _, _, files in os.walk(os.path.join(capture_path, dir_graph_repre, graph_type)):
                            for file in files:
                                try:
                                    size += os.path.getsize(os.path.join(
                                        capture_path, dir_graph_repre, graph_type, file))
                                except:
                                    pass
        dimensions.append(size)

    order_indx = np.argsort(dimensions)
    result_rules_paths_ordered = [result_paths[i] for i in order_indx]

    order_indx = np.argsort(dimensions)
    result_rules_paths_ordered = [result_paths[i] for i in order_indx]
    representatios = []
    print(result_paths)
    for result_path in result_rules_paths_ordered:
        print(f"Capture {result_path}")
        nodes_indx = 0
        capture_name = result_path.split('/')[-1]
        if os.path.isdir(result_path):            
            representatios = ["etdg", "tdg"]
            #Per 10k
            #representatios = ["tdg"]
            print(representatios)
            graph_types = ["full_benign", "full_malicious", "mixed"]
            IP_PORT_MAT_INDX_MAP[result_path.split('/')[-1]] = dict()
            for representation in representatios:
                for graph_type in graph_types:
                    graph_pkl_paths = glob.glob(
                        f"{result_path}/{representation}_graph/{graph_type}/graph_*.pkl")
                    for graph_path in tqdm(graph_pkl_paths):
                        # print(f"{graph_path.split('/')[-1]}")
                        data = open_graph_pkl(file_path=graph_path,
                                              capture_name=result_path.split('/')[-1])
                        node_id_to_port_ip = data['node_id_to_port_ip']
                        for ip_port in node_id_to_port_ip.values():
                            if IP_PORT_MAT_INDX_MAP[capture_name].get(str(ip_port)) is None:
                                # we find a novel distinct ip-port combination
                                IP_PORT_MAT_INDX_MAP[capture_name][str(
                                    ip_port)] = nodes_indx
                                nodes_indx += 1

            # print(IP_PORT_MAT_INDX_MAP)
            # create graphs with adjacence matrix that contains all the nodes
            # result_path, representation, graph_type, capture_name
            # f = functools.partial(add_full_adj,
            #                       result_path,
            #                       representation,
            #                       graph_type,
            #                       capture_name
            #                       )

            # with Pool(1) as p:
            #     p.map(f, graph_pkl_paths)
            for representation in representatios:
                for graph_type in graph_types:
                    graph_pkl_paths = glob.glob(
                        f"{result_path}/{representation}_graph/{graph_type}/graph_*.pkl")
                    for graph_path in tqdm(graph_pkl_paths):
                        add_full_adj(result_path=result_path,
                                     representation=representation,
                                     graph_type=graph_type,
                                     capture_name=capture_name,
                                     graph_path=graph_path)

            # save the map ip-port matrix indx in a json file
            save_file = os.path.join(
                result_path,
                "ip_port_indx_map.json")
            with open(save_file, "w") as outfile:
                print(f"Dumping {save_file}")
                json.dump(IP_PORT_MAT_INDX_MAP[capture_name], outfile)

            print(f"Pre-processing Completed: {result_path.split('/')[-1]}")
