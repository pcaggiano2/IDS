import os
import pandas as pd
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from math import ceil
import json
import functools
# from concurrent.futures import ProcessPoolExecutor as Pool
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing.pool
import time
import math

REP_LIST = ["TDG", "eTDG", "SIM", "ALL"]

def create_similarity_graph(snapshot_data, snapshot_features, interval_number=None, k=5):
    features_matrix = np.zeros(
        (snapshot_data.shape[0], snapshot_features.shape[1]), dtype=float)
    adj_matrix = np.zeros(
        (snapshot_data.shape[0], snapshot_data.shape[0]), dtype=np.uint8)
    labels_matrix = np.zeros((snapshot_data.shape[0], 1), dtype=np.uint8)
    similarity_mat = np.full(
        (snapshot_data.shape[0], snapshot_features.shape[0]), dtype=float, fill_value=-1)
    node_list1 = []
    node_list2 = []
    #  tqdm(range(snapshot_features.shape[0]), desc="General Graph"+str(interval_number), leave=False):
    for flow_id1 in range(snapshot_features.shape[0]):
        features1 = snapshot_features[flow_id1]
        similarity_list = []
        for flow_id2 in range(snapshot_features.shape[0]):
            if flow_id1 != flow_id2:
                features2 = snapshot_features[flow_id2]
                if similarity_mat[flow_id1][flow_id2] < 0:
                    cos_sim = np.dot(
                        features1, features2)/(np.linalg.norm(features1)*np.linalg.norm(features2))
                    similarity_mat[flow_id1][flow_id2] = cos_sim
                    similarity_mat[flow_id2][flow_id1] = cos_sim
                    similarity_list.append((cos_sim, flow_id2))
                else:
                    similarity_list.append(
                        (similarity_mat[flow_id1][flow_id2], flow_id2))
        topk_similarity = sorted(
            similarity_list, key=lambda x: x[0], reverse=True)[:k]
        for _, flow_id2 in topk_similarity:
            adj_matrix[flow_id1][flow_id2] = 1
            node_list1.append(flow_id1)
            node_list2.append(flow_id2)
        features_matrix[flow_id1] = features1
        if snapshot_data.iloc[flow_id1].get("detection_label") is not None:
            labels_matrix[flow_id1] = 0 if snapshot_data.iloc[flow_id1]["detection_label"] == "Benign" else 1
        else:
            labels_matrix[flow_id1] = 0

        node_list1.append(flow_id1)
        node_list2.append(flow_id1)
    node_array1 = np.array(node_list1, dtype=np.uint32)
    node_array2 = np.array(node_list2, dtype=np.uint32)
    return features_matrix, adj_matrix, labels_matrix, (node_array1, node_array2)


def create_trajectory_graph(snapshot_data, snapshot_mat, snapshot_features, interval_number, log_file):
    snapshot_flows = snapshot_data.drop(columns=snapshot_data.columns.difference([
                                        'src_ip', 'dst_ip']),
                                        axis=1).to_numpy()
    features_matrix = np.zeros(
        (snapshot_data.shape[0], snapshot_features.shape[1]), dtype=float)
    adj_matrix = np.zeros(
        (snapshot_data.shape[0], snapshot_data.shape[0]), dtype=np.uint8)
    labels_matrix = np.zeros((snapshot_data.shape[0], 1), dtype=np.uint8)
    node_list1 = []
    node_list2 = []
    # tqdm(range(snapshot_flows.shape[0]), desc="Trajectory Graph "+str(interval_number), leave=False):
    for flow_id1 in range(snapshot_flows.shape[0]):
        flow1 = snapshot_flows[flow_id1]
        for flow_id2 in range(snapshot_flows.shape[0]):
            if flow_id1 != flow_id2:
                flow2 = snapshot_flows[flow_id2]
                if flow1[0] == flow2[0] or flow1[1] == flow2[1]:
                    adj_matrix[flow_id1][flow_id2] = 1
                    node_list1.append(flow_id1)
                    node_list2.append(flow_id2)
            else:
                node_list1.append(flow_id1)
                node_list2.append(flow_id1)
        features_matrix[flow_id1] = snapshot_features[flow_id1]
        if snapshot_data.iloc[flow_id1].get("detection_label") is not None:
            labels_matrix[flow_id1] = 0 if snapshot_data.iloc[flow_id1]["detection_label"] == "Benign" else 1
        else:
            labels_matrix[flow_id1] = 0

    node_array1 = np.array(node_list1, dtype=np.uint32)
    node_array2 = np.array(node_list2, dtype=np.uint32)

    # tqdm(range(adj_matrix.shape[0]), desc="Verify traj "+str(interval_number), leave=False):
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[0]):
            if i != j:
                flow1 = snapshot_mat[i]
                flow2 = snapshot_mat[j]
                if adj_matrix[i][j] == 1:
                    if not (flow1[2] == flow2[2] or flow1[6] == flow2[6]):
                        log_file.write(
                            " Wrong edge ("+str(i)+", "+str(j)+")\n")
                else:
                    if (flow1[2] == flow2[2] or flow1[6] == flow2[6]):
                        log_file.write(
                            " Missing edge ("+str(i)+", "+str(j)+")\n")
    return features_matrix, adj_matrix, labels_matrix, (node_array1, node_array2)


def to_global_graph_dict(global_dict, cc_dict):
    for key, value in cc_dict.items():
        global_dict[key] = value


def to_global_graph_dict_degree(global_dict, cc_node_degree):
    for node, degree in cc_node_degree:
        global_dict[node] = degree


def is_barycenter(barycenter_dict, barycenter_nodes, nodes):
    for node in nodes:
        barycenter_dict[node] = 1 if node in barycenter_nodes else 0


def is_center(barycenter_dict, barycenter_nodes, nodes):
    for node in nodes:
        barycenter_dict[node] = 1 if node in barycenter_nodes else 0


def to_node_betweenness(global_dict, edge_betweenness):
    for edge, betweenness in edge_betweenness.items():
        src_node, dst_node = edge
        if src_node not in global_dict:
            global_dict[src_node] = betweenness
        else:
            global_dict[src_node] = betweenness
        if dst_node not in global_dict:
            global_dict[dst_node] = betweenness
        else:
            global_dict[dst_node] = betweenness


def create_tdg(snapshot_mat, snapshot_features, label):
    """Create tdg graph without structural features

    Args:
        snapshot_mat (_type_): _description_
        snapshot_features (_type_): _description_
        log_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    id = 0
    graph_dict = {}  # this dict keeps track of all node-id, features, label
    node_ids_dict = {}  # this dict maps (ip, port) -> node_id
    node_reverse_dict = {}  # this dict maps node_id -> (ip, port)
    # tqdm(range(snapshot_mat.shape[0]), desc="Extended find nodes", leave=False):
    for i in range(snapshot_mat.shape[0]):
        flow = snapshot_mat[i]
        src_ip = flow[2]
        src_port = flow[5]
        dst_ip = flow[6]
        dst_port = flow[9]
        src_node = (src_ip, src_port)
        dst_node = (dst_ip, dst_port)

        if src_node not in node_ids_dict:
            node_ids_dict[src_node] = id
            node_reverse_dict[id] = src_node
            id += 1

        if dst_node not in node_ids_dict:
            node_ids_dict[dst_node] = id
            node_reverse_dict[id] = dst_node
            id += 1

        src_node_id = node_ids_dict[src_node]
        dst_node_id = node_ids_dict[dst_node]

        if label:
            if src_node_id not in graph_dict:
                graph_dict[src_node_id] = [
                    (dst_node_id, snapshot_features[i], 0 if flow[-2] == "Benign" else 1)]
            else:
                graph_dict[src_node_id].append(
                    (dst_node_id, snapshot_features[i], 0 if flow[-2] == "Benign" else 1))
            if dst_node_id not in graph_dict:
                graph_dict[dst_node_id] = [
                    (src_node_id, snapshot_features[i], 0 if flow[-2] == "Benign" else 1)] 
            else:
                graph_dict[dst_node_id].append(
                    (src_node_id, snapshot_features[i], 0 if flow[-2] == "Benign" else 1))
            
        else:
            if src_node_id not in graph_dict:
                graph_dict[src_node_id] = [
                    (dst_node_id, snapshot_features[i], 0)]
            else:
                graph_dict[src_node_id].append(
                    (dst_node_id, snapshot_features[i], 0))

            if dst_node_id not in graph_dict:
                graph_dict[dst_node_id] = [
                    (src_node_id, snapshot_features[i], 0)]
            else:
                graph_dict[dst_node_id].append(
                    (src_node_id, snapshot_features[i], 0))

    total_nodes = len(node_ids_dict.keys())
    print(f"Total nodes {total_nodes}")
    flow_features_matrix = np.zeros(
        (total_nodes, snapshot_features.shape[1]), dtype=float)
    node_features_matrix = np.zeros(
        (total_nodes, flow_features_matrix.shape[1]), dtype=float)
    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=np.uint8)
    edges_list = []
    vis_edges_list = []
    labels_matrix = np.zeros((total_nodes,), dtype=np.uint8)
    flow_label_matrix = np.zeros((total_nodes, total_nodes), dtype=np.uint8)

    node_list1 = []
    node_list2 = []
    ids_tuple = list(node_ids_dict.items())

    # for each node in the graph
    # tqdm(range(len(ids_tuple)), desc="Extendend graph", leave=False):
    for i in range(len(ids_tuple)):
        _, src_node_id = ids_tuple[i]
        edge_number = 0
        benign_edge_number = 0
        edge_features_sum = np.zeros(
            (snapshot_features.shape[1],), dtype=float)

        for dst_node_id, edge_features, edge_label in graph_dict[src_node_id]:
            edge_features_sum += np.array(edge_features, dtype=float)
            adj_matrix[src_node_id][dst_node_id] = 1
            flow_label_matrix[src_node_id][dst_node_id] = edge_label
            node_list1.append(src_node_id)
            node_list2.append(dst_node_id)
            src_node = node_reverse_dict[src_node_id]
            dst_node = node_reverse_dict[dst_node_id]
            vis_edges_list.append(
                (src_node[0]+":"+str(src_node[1]), dst_node[0]+":"+str(dst_node[1])))
            edges_list.append((src_node_id, dst_node_id))
            edges_list.append((src_node_id, dst_node_id, edge_label))
            
            benign_edge_number += 1 if edge_label == 0 else 0
            edge_number += 1
        node_features = edge_features_sum/edge_number
        flow_features_matrix[src_node_id] = node_features
        labels_matrix[src_node_id] = 0 if benign_edge_number > edge_number/2 else 1
        node_list1.append(src_node_id)
        node_list2.append(src_node_id)

    node_array1 = np.array(node_list1, dtype=np.uint32)
    node_array2 = np.array(node_list2, dtype=np.uint32)

    for i in range(node_features_matrix.shape[0]):
        node_features_matrix[i] = flow_features_matrix[i]
    return node_features_matrix, adj_matrix, labels_matrix, (node_array1, node_array2), node_reverse_dict, edges_list, node_ids_dict, flow_label_matrix


def create_extendend_graph(snapshot_mat, snapshot_features, log_file, edges_list, node_ids_dict, flow_features_matrix, adj_matrix, labels_matrix):
    """_summary_

    Args:
        snapshot_mat (_type_): _description_
        snapshot_features (_type_): _description_
        log_file (_type_): _description_
        edges_list (_type_): _description_
        node_ids_dict (_type_): _description_
        flow_features_matrix (_type_): _description_
        adj_matrix (_type_): _description_
        labels_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """

    total_nodes = len(node_ids_dict.keys())
    structutal_features_matrix = np.zeros((total_nodes, 15), dtype=float)
    node_features_matrix = np.zeros(
        (total_nodes, flow_features_matrix.shape[1]+structutal_features_matrix.shape[1]), dtype=float)

    graph = nx.Graph(edges_list)
    graph_degree_centrality = nx.degree_centrality(graph)

    graph_betweenness_centrality = nx.betweenness_centrality(graph)
    graph_closeness_centrality = nx.closeness_centrality(graph)

    cc_degree_centrality = {}
    cc_betweenness_centrality = {}
    cc_closeness_centrality = {}
    second_order_centrality = {}
    edge_betweenness_centrality = {}
    eccentricity = {}
    barycenter = {}
    radius = {}
    center = {}
    size = {}
    degree = {}
    degree_two_hops = {}
    connected_components = [graph.subgraph(
        c).copy() for c in nx.connected_components(graph)]
    # tqdm(, desc="Structural features", leave=False):
    for i in range(len(connected_components)):
        cc = connected_components[i]
        cc_eccentricity = nx.eccentricity(cc)
        cc_radius = nx.radius(cc, e=cc_eccentricity)
        to_global_graph_dict(cc_degree_centrality, nx.degree_centrality(cc))
        to_global_graph_dict(cc_betweenness_centrality,
                             nx.betweenness_centrality(cc))
        to_global_graph_dict(cc_closeness_centrality,
                             nx.closeness_centrality(cc))
        to_global_graph_dict(second_order_centrality,
                             nx.second_order_centrality(cc))
        to_global_graph_dict(eccentricity, cc_eccentricity)
        to_node_betweenness(edge_betweenness_centrality,
                            nx.edge_betweenness_centrality(cc))
        to_global_graph_dict_degree(degree, cc.degree())

        is_barycenter(barycenter, nx.barycenter(cc), cc.nodes)
        is_center(center, nx.center(cc, e=cc_eccentricity), cc.nodes)
        for node in cc.nodes:
            degree_two_hops[node] = len(nx.ego_graph(
                cc, node, radius=2, undirected=True).nodes())
            size[node] = cc.number_of_nodes()
            radius[node] = cc_radius

    for _, node_id in node_ids_dict.items():
        structutal_features_matrix[node_id][0] = graph_degree_centrality[node_id]
        structutal_features_matrix[node_id][1] = graph_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][2] = graph_closeness_centrality[node_id]
        structutal_features_matrix[node_id][3] = cc_degree_centrality[node_id]
        structutal_features_matrix[node_id][4] = cc_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][5] = cc_closeness_centrality[node_id]
        structutal_features_matrix[node_id][6] = second_order_centrality[node_id]
        structutal_features_matrix[node_id][7] = edge_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][8] = eccentricity[node_id]
        structutal_features_matrix[node_id][9] = barycenter[node_id]
        structutal_features_matrix[node_id][10] = radius[node_id]
        structutal_features_matrix[node_id][11] = center[node_id]
        structutal_features_matrix[node_id][12] = size[node_id]
        structutal_features_matrix[node_id][13] = degree[node_id]
        structutal_features_matrix[node_id][14] = degree_two_hops[node_id]

    for i in range(node_features_matrix.shape[0]):
        node_features_matrix[i] = np.concatenate(
            [flow_features_matrix[i], structutal_features_matrix[i]])

    # tqdm(), desc="Verify extedend", leave=False):
    for i in range(snapshot_mat.shape[0]):
        flow = snapshot_mat[i]
        src_ip = flow[2]
        src_port = flow[5]
        dst_ip = flow[6]
        dst_port = flow[9]
        src_node = (src_ip, src_port)
        dst_node = (dst_ip, dst_port)
        src_node_id = node_ids_dict[src_node]
        dst_node_id = node_ids_dict[dst_node]
        if adj_matrix[src_node_id][dst_node_id] != 1:
            log_file.write(" Missing edge ("+str(src_node_id) +
                           ", "+str(dst_node_id)+")\n")

    return node_features_matrix, adj_matrix, labels_matrix, edges_list

def create_extendend_graph_approx_50(snapshot_mat, snapshot_features, log_file, edges_list, node_ids_dict, flow_features_matrix, adj_matrix, labels_matrix):
    """_summary_

    Args:
        snapshot_mat (_type_): _description_
        snapshot_features (_type_): _description_
        log_file (_type_): _description_
        edges_list (_type_): _description_
        node_ids_dict (_type_): _description_
        flow_features_matrix (_type_): _description_
        adj_matrix (_type_): _description_
        labels_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """

    total_nodes = len(node_ids_dict.keys())
    structutal_features_matrix = np.zeros((total_nodes, 15), dtype=float)
    node_features_matrix = np.zeros(
        (total_nodes, flow_features_matrix.shape[1]+structutal_features_matrix.shape[1]), dtype=float)

    graph = nx.Graph(edges_list)
    graph_degree_centrality = nx.degree_centrality(graph)
    x = graph.number_of_nodes()
    graph_betweenness_centrality = nx.betweenness_centrality(graph, k = math.floor(x * 0.5))
    graph_closeness_centrality = nx.closeness_centrality(graph)

    cc_degree_centrality = {}
    cc_betweenness_centrality = {}
    cc_closeness_centrality = {}
    second_order_centrality = {}
    edge_betweenness_centrality = {}
    eccentricity = {}
    barycenter = {}
    radius = {}
    center = {}
    size = {}
    degree = {}
    degree_two_hops = {}

    connected_components = [graph.subgraph(
        c).copy() for c in nx.connected_components(graph)]
    # tqdm(, desc="Structural features", leave=False):
    for i in range(len(connected_components)):
        cc = connected_components[i]
        rff = math.floor(cc.number_of_nodes() * 0.50 )
        cc_eccentricity = nx.eccentricity(cc)
        cc_radius = nx.radius(cc, e=cc_eccentricity)
        to_global_graph_dict(cc_degree_centrality, nx.degree_centrality(cc))
        to_global_graph_dict(cc_betweenness_centrality,
                             nx.betweenness_centrality(cc, k = rff))
        to_global_graph_dict(cc_closeness_centrality,
                             nx.closeness_centrality(cc))
        to_global_graph_dict(second_order_centrality,
                             nx.second_order_centrality(cc))
        to_global_graph_dict(eccentricity, cc_eccentricity)
        to_node_betweenness(edge_betweenness_centrality,
                            nx.edge_betweenness_centrality(cc, k = rff))
        to_global_graph_dict_degree(degree, cc.degree())

        is_barycenter(barycenter, nx.barycenter(cc), cc.nodes)
        is_center(center, nx.center(cc, e=cc_eccentricity), cc.nodes)
        for node in cc.nodes:
            degree_two_hops[node] = len(nx.ego_graph(
                cc, node, radius=2, undirected=True).nodes())
            size[node] = cc.number_of_nodes()
            radius[node] = cc_radius

    for _, node_id in node_ids_dict.items():
        structutal_features_matrix[node_id][0] = graph_degree_centrality[node_id]
        structutal_features_matrix[node_id][1] = graph_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][2] = graph_closeness_centrality[node_id]
        structutal_features_matrix[node_id][3] = cc_degree_centrality[node_id]
        structutal_features_matrix[node_id][4] = cc_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][5] = cc_closeness_centrality[node_id]
        structutal_features_matrix[node_id][6] = second_order_centrality[node_id]
        structutal_features_matrix[node_id][7] = edge_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][8] = eccentricity[node_id]
        structutal_features_matrix[node_id][9] = barycenter[node_id]
        structutal_features_matrix[node_id][10] = radius[node_id]
        structutal_features_matrix[node_id][11] = center[node_id]
        structutal_features_matrix[node_id][12] = size[node_id]
        structutal_features_matrix[node_id][13] = degree[node_id]
        structutal_features_matrix[node_id][14] = degree_two_hops[node_id]

    for i in range(node_features_matrix.shape[0]):
        node_features_matrix[i] = np.concatenate(
            [flow_features_matrix[i], structutal_features_matrix[i]])

    # tqdm(), desc="Verify extedend", leave=False):
    for i in range(snapshot_mat.shape[0]):
        flow = snapshot_mat[i]
        src_ip = flow[2]
        src_port = flow[5]
        dst_ip = flow[6]
        dst_port = flow[9]
        src_node = (src_ip, src_port)
        dst_node = (dst_ip, dst_port)
        src_node_id = node_ids_dict[src_node]
        dst_node_id = node_ids_dict[dst_node]
        if adj_matrix[src_node_id][dst_node_id] != 1:
            log_file.write(" Missing edge ("+str(src_node_id) +
                           ", "+str(dst_node_id)+")\n")

    return node_features_matrix, adj_matrix, labels_matrix, edges_list

def create_extendend_graph_approx_75(snapshot_mat, snapshot_features, log_file, edges_list, node_ids_dict, flow_features_matrix, adj_matrix, labels_matrix):
    """_summary_

    Args:
        snapshot_mat (_type_): _description_
        snapshot_features (_type_): _description_
        log_file (_type_): _description_
        edges_list (_type_): _description_
        node_ids_dict (_type_): _description_
        flow_features_matrix (_type_): _description_
        adj_matrix (_type_): _description_
        labels_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """

    total_nodes = len(node_ids_dict.keys())
    structutal_features_matrix = np.zeros((total_nodes, 15), dtype=float)
    node_features_matrix = np.zeros(
        (total_nodes, flow_features_matrix.shape[1]+structutal_features_matrix.shape[1]), dtype=float)

    graph = nx.Graph(edges_list)
    graph_degree_centrality = nx.degree_centrality(graph)
    graph_betweenness_centrality = nx.betweenness_centrality(graph, k = math.floor(graph.number_of_nodes() * 0.75))
    graph_closeness_centrality = nx.closeness_centrality(graph)
    cc_degree_centrality = {}
    cc_betweenness_centrality = {}
    cc_closeness_centrality = {}
    second_order_centrality = {}
    edge_betweenness_centrality = {}
    eccentricity = {}
    barycenter = {}
    radius = {}
    center = {}
    size = {}
    degree = {}
    degree_two_hops = {}

    connected_components = [graph.subgraph(
        c).copy() for c in nx.connected_components(graph)]
    # tqdm(, desc="Structural features", leave=False):
    for i in range(len(connected_components)):
        cc = connected_components[i]
        rff = math.floor(cc.number_of_nodes() * 0.75 )
        cc_eccentricity = nx.eccentricity(cc)
        cc_radius = nx.radius(cc, e=cc_eccentricity)
        to_global_graph_dict(cc_degree_centrality, nx.degree_centrality(cc))
        to_global_graph_dict(cc_betweenness_centrality,
                             nx.betweenness_centrality(cc, k = rff))
        to_global_graph_dict(cc_closeness_centrality,
                             nx.closeness_centrality(cc))
        to_global_graph_dict(second_order_centrality,
                             nx.second_order_centrality(cc))
        to_global_graph_dict(eccentricity, cc_eccentricity)
        to_node_betweenness(edge_betweenness_centrality,
                            nx.edge_betweenness_centrality(cc, k = rff))
        to_global_graph_dict_degree(degree, cc.degree())

        is_barycenter(barycenter, nx.barycenter(cc), cc.nodes)
        is_center(center, nx.center(cc, e=cc_eccentricity), cc.nodes)
        for node in cc.nodes:
            degree_two_hops[node] = len(nx.ego_graph(
                cc, node, radius=2, undirected=True).nodes())
            size[node] = cc.number_of_nodes()
            radius[node] = cc_radius

    for _, node_id in node_ids_dict.items():
        structutal_features_matrix[node_id][0] = graph_degree_centrality[node_id]
        structutal_features_matrix[node_id][1] = graph_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][2] = graph_closeness_centrality[node_id]
        structutal_features_matrix[node_id][3] = cc_degree_centrality[node_id]
        structutal_features_matrix[node_id][4] = cc_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][5] = cc_closeness_centrality[node_id]
        structutal_features_matrix[node_id][6] = second_order_centrality[node_id]
        structutal_features_matrix[node_id][7] = edge_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][8] = eccentricity[node_id]
        structutal_features_matrix[node_id][9] = barycenter[node_id]
        structutal_features_matrix[node_id][10] = radius[node_id]
        structutal_features_matrix[node_id][11] = center[node_id]
        structutal_features_matrix[node_id][12] = size[node_id]
        structutal_features_matrix[node_id][13] = degree[node_id]
        structutal_features_matrix[node_id][14] = degree_two_hops[node_id]

    for i in range(node_features_matrix.shape[0]):
        node_features_matrix[i] = np.concatenate(
            [flow_features_matrix[i], structutal_features_matrix[i]])

    # tqdm(), desc="Verify extedend", leave=False):
    for i in range(snapshot_mat.shape[0]):
        flow = snapshot_mat[i]
        src_ip = flow[2]
        src_port = flow[5]
        dst_ip = flow[6]
        dst_port = flow[9]
        src_node = (src_ip, src_port)
        dst_node = (dst_ip, dst_port)
        src_node_id = node_ids_dict[src_node]
        dst_node_id = node_ids_dict[dst_node]
        if adj_matrix[src_node_id][dst_node_id] != 1:
            log_file.write(" Missing edge ("+str(src_node_id) +
                           ", "+str(dst_node_id)+")\n")

    return node_features_matrix, adj_matrix, labels_matrix, edges_list


def create_extendend_graph_approx_60(snapshot_mat, snapshot_features, log_file, edges_list, node_ids_dict, flow_features_matrix, adj_matrix, labels_matrix):
    """_summary_

    Args:
        snapshot_mat (_type_): _description_
        snapshot_features (_type_): _description_
        log_file (_type_): _description_
        edges_list (_type_): _description_
        node_ids_dict (_type_): _description_
        flow_features_matrix (_type_): _description_
        adj_matrix (_type_): _description_
        labels_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """

    total_nodes = len(node_ids_dict.keys())
    structutal_features_matrix = np.zeros((total_nodes, 15), dtype=float)
    node_features_matrix = np.zeros(
        (total_nodes, flow_features_matrix.shape[1]+structutal_features_matrix.shape[1]), dtype=float)

    graph = nx.Graph(edges_list)
    graph_degree_centrality = nx.degree_centrality(graph)
    graph_betweenness_centrality = nx.betweenness_centrality(graph, k = math.floor(graph.number_of_nodes() * 0.60))
    graph_closeness_centrality = nx.closeness_centrality(graph)

    cc_degree_centrality = {}
    cc_betweenness_centrality = {}
    cc_closeness_centrality = {}
    second_order_centrality = {}
    edge_betweenness_centrality = {}
    eccentricity = {}
    barycenter = {}
    radius = {}
    center = {}
    size = {}
    degree = {}
    degree_two_hops = {}

    connected_components = [graph.subgraph(
        c).copy() for c in nx.connected_components(graph)]
    # tqdm(, desc="Structural features", leave=False):
    for i in range(len(connected_components)):
        cc = connected_components[i]
        rff = math.floor(cc.number_of_nodes() * 0.60)
        cc_eccentricity = nx.eccentricity(cc)
        cc_radius = nx.radius(cc, e=cc_eccentricity)
        to_global_graph_dict(cc_degree_centrality, nx.degree_centrality(cc))
        to_global_graph_dict(cc_betweenness_centrality,
                             nx.betweenness_centrality(cc, k = rff))
        to_global_graph_dict(cc_closeness_centrality,
                             nx.closeness_centrality(cc))
        to_global_graph_dict(second_order_centrality,
                             nx.second_order_centrality(cc))
        to_global_graph_dict(eccentricity, cc_eccentricity)
        to_node_betweenness(edge_betweenness_centrality,
                            nx.edge_betweenness_centrality(cc, k = rff))
        to_global_graph_dict_degree(degree, cc.degree())

        is_barycenter(barycenter, nx.barycenter(cc), cc.nodes)
        is_center(center, nx.center(cc, e=cc_eccentricity), cc.nodes)
        for node in cc.nodes:
            degree_two_hops[node] = len(nx.ego_graph(
                cc, node, radius=2, undirected=True).nodes())
            size[node] = cc.number_of_nodes()
            radius[node] = cc_radius

    for _, node_id in node_ids_dict.items():
        structutal_features_matrix[node_id][0] = graph_degree_centrality[node_id]
        structutal_features_matrix[node_id][1] = graph_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][2] = graph_closeness_centrality[node_id]
        structutal_features_matrix[node_id][3] = cc_degree_centrality[node_id]
        structutal_features_matrix[node_id][4] = cc_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][5] = cc_closeness_centrality[node_id]
        structutal_features_matrix[node_id][6] = second_order_centrality[node_id]
        structutal_features_matrix[node_id][7] = edge_betweenness_centrality[node_id]
        structutal_features_matrix[node_id][8] = eccentricity[node_id]
        structutal_features_matrix[node_id][9] = barycenter[node_id]
        structutal_features_matrix[node_id][10] = radius[node_id]
        structutal_features_matrix[node_id][11] = center[node_id]
        structutal_features_matrix[node_id][12] = size[node_id]
        structutal_features_matrix[node_id][13] = degree[node_id]
        structutal_features_matrix[node_id][14] = degree_two_hops[node_id]

    for i in range(node_features_matrix.shape[0]):
        node_features_matrix[i] = np.concatenate(
            [flow_features_matrix[i], structutal_features_matrix[i]])

    # tqdm(), desc="Verify extedend", leave=False):
    for i in range(snapshot_mat.shape[0]):
        flow = snapshot_mat[i]
        src_ip = flow[2]
        src_port = flow[5]
        dst_ip = flow[6]
        dst_port = flow[9]
        src_node = (src_ip, src_port)
        dst_node = (dst_ip, dst_port)
        src_node_id = node_ids_dict[src_node]
        dst_node_id = node_ids_dict[dst_node]
        if adj_matrix[src_node_id][dst_node_id] != 1:
            log_file.write(" Missing edge ("+str(src_node_id) +
                           ", "+str(dst_node_id)+")\n")

    return node_features_matrix, adj_matrix, labels_matrix, edges_list


def split_dataset(dataset):
    negative_data = []
    positive_data = []
    for i in range(dataset.shape[0]):
        if dataset[i][-2] == "Benign":
            negative_data.append(dataset[i])
        else:
            positive_data.append(dataset[i])
    return np.array(negative_data), np.array(positive_data)


def count_positve(dataset):

    return (dataset['detection_label'] != 'Benign').sum()


def save_file(out_path, capture_name, graph_type, graph, full_benign, mixed, full_malicious, interval_number):
    # save file
    if full_benign:
        save_file_path = os.path.join(os.path.join(out_path,
                                                   capture_name,
                                                   graph_type, "full_benign", f"graph_{interval_number}.pkl"))
        print("Save full benign")
    elif mixed:
        save_file_path = os.path.join(os.path.join(out_path,
                                                   capture_name,
                                                   graph_type, "mixed", f"graph_{interval_number}.pkl"))
        print("Save mixed")

    elif full_malicious:
        save_file_path = os.path.join(os.path.join(out_path,
                                                   capture_name,
                                                   graph_type, "full_malicious", f"graph_{interval_number}.pkl"))
        print("Save full malicious")

    with open(save_file_path, "wb") as handle:
        pickle.dump(graph, handle, protocol=4)


def create_graph(out_path=None, capture_name=None, graph_type="tdg", snapshot_data=None, snapshot_features=None, interval_number=None, stats=None, log_file=None, full_benign=False, mixed=False, full_malicious=False, also_etdg=False):
    label = True
    if len(snapshot_data.get("detection_label", [])) == 0:
        label = False

    stats[graph_type] = dict()

    if graph_type == "tdg_graph" or graph_type == "etdg_graph":
        start = time.time()
        features_matrix, adj_matrix, labels_matrix, edge_list, node_id_to_port_ip, edges_list, node_ids_dict, flow_label_matrix = create_tdg(
            snapshot_mat=snapshot_data.to_numpy(),
            snapshot_features=snapshot_features,
            label=label)
        graph = {"node_features": features_matrix, "adj": adj_matrix,
                 "node_labels": labels_matrix, "edge_list": edges_list,
                 "node_id_to_port_ip": node_id_to_port_ip,
                 "flow_label_matrix": flow_label_matrix}
        
        print('Save_TDG')
        save_file(out_path=out_path,
                    capture_name=capture_name,
                    graph_type="tdg_graph",
                    graph=graph,
                    full_benign=full_benign,
                    full_malicious=full_malicious,
                    mixed=mixed,
                    interval_number=interval_number)
        print("Finish Save TDG")

        if graph_type == "etdg_graph":
            print('Also_eTDG_processing')
            extended_graph_log_filename = out_path+ capture_name +"/" + capture_name + ".txt"
            extended_graph_log_file = open(extended_graph_log_filename,"a")
            # snapshot_mat, snapshot_features, log_file, edges_list, node_ids_dict, flow_features_matrix, adj_matrix, labels_matrix
            stats["etdg_graph"] = dict()
            if "50_perc" in extended_graph_log_filename:
                start = time.time()
                features_matrix_ex, _, _, _ = create_extendend_graph_approx_50(
                    snapshot_mat=snapshot_data.to_numpy(),
                    snapshot_features=snapshot_features,
                    log_file=extended_graph_log_file,
                    edges_list=edges_list,
                    node_ids_dict=node_ids_dict,
                    labels_matrix=labels_matrix,
                    adj_matrix=adj_matrix,
                    flow_features_matrix=features_matrix)
                delta = time.time() - start
            elif "75_perc" in extended_graph_log_filename:
                start = time.time()
                features_matrix_ex, _, _, _ = create_extendend_graph_approx_75(
                    snapshot_mat=snapshot_data.to_numpy(),
                    snapshot_features=snapshot_features,
                    log_file=extended_graph_log_file,
                    edges_list=edges_list,
                    node_ids_dict=node_ids_dict,
                    labels_matrix=labels_matrix,
                    adj_matrix=adj_matrix,
                    flow_features_matrix=features_matrix)
                delta = time.time() - start
            elif "60_perc" in extended_graph_log_filename:
                start = time.time()
                features_matrix_ex, _, _, _ = create_extendend_graph_approx_60(
                    snapshot_mat=snapshot_data.to_numpy(),
                    snapshot_features=snapshot_features,
                    log_file=extended_graph_log_file,
                    edges_list=edges_list,
                    node_ids_dict=node_ids_dict,
                    labels_matrix=labels_matrix,
                    adj_matrix=adj_matrix,
                    flow_features_matrix=features_matrix)
                delta = time.time() - start
            else:
                start = time.time()
                features_matrix_ex, _, _, _ = create_extendend_graph(
                    snapshot_mat=snapshot_data.to_numpy(),
                    snapshot_features=snapshot_features,
                    log_file=extended_graph_log_file,
                    edges_list=edges_list,
                    node_ids_dict=node_ids_dict,
                    labels_matrix=labels_matrix,
                    adj_matrix=adj_matrix,
                    flow_features_matrix=features_matrix)
                delta = time.time() - start
            graph = {"node_features": features_matrix_ex, "adj": adj_matrix,
                     "node_labels": labels_matrix, "edge_list": edge_list,
                     "node_id_to_port_ip": node_id_to_port_ip}
            print('Save_eTDG')
            save_file(out_path=out_path,
                      capture_name=capture_name,
                      graph_type="etdg_graph",
                      graph=graph,
                      full_benign=full_benign,
                      full_malicious=full_malicious,
                      mixed=mixed,
                      interval_number=interval_number)
            
            number_nodes = adj_matrix.shape[0]
            number_edges = np.sum(np.sum(adj_matrix))
            number_flows = snapshot_data.shape[0]

            # save stats
            stats["etdg_graph"][f"graph_{interval_number}"] = dict()
            # save number of nodes, edges, flows and type
            stats["etdg_graph"][f"graph_{interval_number}"]["nodes"] = float(
                number_nodes)
            stats["etdg_graph"][f"graph_{interval_number}"]["edges"] = float(
                number_edges)
            stats["etdg_graph"][f"graph_{interval_number}"]["flows"] = float(
                number_flows)
            if full_benign:
                stats["etdg_graph"][f"graph_{interval_number}"]["type"] = 1.0
            elif mixed:
                stats["etdg_graph"][f"graph_{interval_number}"]["type"] = 2.0
            else:
                stats["etdg_graph"][f"graph_{interval_number}"]["type"] = 3.0

    else:
        features_matrix, adj_matrix, labels_matrix, edge_list  = create_similarity_graph(
        snapshot_data=snapshot_data,
        snapshot_features=snapshot_features)
        graph = {"node_features": features_matrix, "adj": adj_matrix,
                "node_labels": labels_matrix, "edge_list": edge_list}
        
        print('Save_SIM')
        save_file(out_path=out_path,
                    capture_name=capture_name,
                    graph_type="sim_graph",
                    graph=graph,
                    full_benign=full_benign,
                    full_malicious=full_malicious,
                    mixed=mixed,
                    interval_number=interval_number)

        number_nodes = adj_matrix.shape[0]
        number_edges = np.sum(np.sum(adj_matrix))
        number_flows = snapshot_data.shape[0]

        # save stats
        stats[graph_type][f"graph_{interval_number}"] = dict()
        # save number of nodes, edges, flows and type
        stats[graph_type][f"graph_{interval_number}"]["nodes"] = float(
            number_nodes)
        stats[graph_type][f"graph_{interval_number}"]["edges"] = float(
            number_edges)
        stats[graph_type][f"graph_{interval_number}"]["flows"] = float(
            number_flows)
        if full_benign:
            stats[graph_type][f"graph_{interval_number}"]["type"] = 1.0
        elif mixed:
            stats[graph_type][f"graph_{interval_number}"]["type"] = 2.0
        else:
            stats[graph_type][f"graph_{interval_number}"]["type"] = 3.0


def generate_graph_from_snapshot(out_path, log_file_path, capture_name, k, features, graph_type, time_indx):
    time_interval, interval_number = time_indx
    log_file = open(log_file_path, 'a')
    stats = dict()

    # check if the graph in this interval has been already created
    exist_sim = False
    exist_traj = False
    exist_etdg = False
    exist_tdg = False
    #for gt in ["similarity_graph", "trajectory_graph", "etdg_graph", "tdg_graph"]:
    if graph_type == 'eTDG' or graph_type == "ALL":
        for gt in ["etdg_graph"]:
            for gl in ["full_benign", "mixed", "full_malicious"]:
                if os.path.exists(os.path.join(os.path.join(out_path,
                                                            capture_name,
                                                            gt, gl, f"graph_{interval_number}.pkl"))):
                    if gt == "etdg_graph":
                        exist_etdg = True

    elif graph_type == 'TDG' or graph_type == "ALL":
        for gt in ["tdg_graph"]:
            for gl in ["full_benign", "mixed", "full_malicious"]:
                if os.path.exists(os.path.join(os.path.join(out_path,
                                                            capture_name,
                                                            gt, gl, f"graph_{interval_number}.pkl"))):
                    if gt == "tdg_graph":
                        exist_tdg = True
    elif graph_type == 'SIM' or graph_type == "ALL":
        for gt in ["sim_graph"]:
            for gl in ["full_benign", "mixed", "full_malicious"]:
                if os.path.exists(os.path.join(os.path.join(out_path,
                                                            capture_name,
                                                            gt, gl, f"graph_{interval_number}.pkl"))):
                    if gt == "sim_graph":
                        exist_sim = True

    start_ts, end_ts = time_interval
    # get data in snapshot
    snapshot_data = features.loc[(features["bidirectional_first_seen_ms"] >= start_ts) & (
        features["bidirectional_first_seen_ms"] < end_ts)]
    number_flows = snapshot_data.shape[0]
    print(f"\nNumber of flows in {number_flows}")

    # identify graph type [full_benign, mixed, full_malicious]
    # Take label column
    no_label = False
    if len(snapshot_data.get("detection_label", [])) != 0:
        flow_type = snapshot_data["detection_label"]
        count_values = flow_type.value_counts()
        if 'Benign' in count_values:
            benign_flows = count_values['Benign']
        else:
            benign_flows = 0
        if 'Malicious' in count_values:
            malicious_flows = count_values['Malicious']
        else:
            malicious_flows = 0
    else:
        benign_flows = number_flows
        malicious_flows = 0
        no_label = True

    # [full_benign, mixed, full_malicious]
    full_benign = False
    mixed = False
    full_malicious = False
    if malicious_flows == 0 and benign_flows == number_flows:
        full_benign = True
    if malicious_flows == number_flows and benign_flows == 0:
        full_malicious = True
    if malicious_flows != 0 and benign_flows != 0:
        mixed = True
    print(
        f"Number benign {benign_flows} - Number malign {malicious_flows}")

    # List of columns to drop, including "attack_label" and "detection_label" conditionally
    columns_to_drop = [
        "id", "expiration_id", "src_ip", "src_mac", "src_oui", "src_port", "dst_ip", "dst_mac", "dst_oui",
        "dst_port", "protocol", "ip_version", "vlan_id", "tunnel_id", "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
        "src2dst_first_seen_ms", "src2dst_last_seen_ms", "dst2src_first_seen_ms", "dst2src_last_seen_ms"
    ]

    bot_iot_flag = False
    #check if string out_path contains as substring "Bot-IoT"
    if "Bot-IoT" in out_path:
        bot_iot_flag = True
        print("Bot-IoT dataset")
        
    if snapshot_data.shape[0] > 0:

        # Limit malicious flows to 20,000 if required
        if bot_iot_flag and malicious_flows > 20000:
            print(f"Reducing malicious flows from {malicious_flows} to 20,000")
            
            # Separate malicious and benign flows
            malicious_data = snapshot_data[snapshot_data["detection_label"] == "Malicious"]
            benign_data = snapshot_data[snapshot_data["detection_label"] == "Benign"]
            
            # Randomly sample 20,000 malicious flows
            malicious_data = malicious_data.sample(n=20000, random_state=42)

            # Combine the sampled malicious data with benign data
            snapshot_data = pd.concat([benign_data, malicious_data])

            # Sort the combined data by 'bidirectional_first_seen_ms'
            snapshot_data.sort_values("bidirectional_first_seen_ms", inplace=True)

            print(f"After sampling and sorting: {snapshot_data.shape[0]} flows "
                f"(Benign: {benign_data.shape[0]}, Malicious: {malicious_data.shape[0]})")


        # remove categorical features
        if not no_label:

            if "detection_label" in snapshot_data.columns:
                columns_to_drop.append("detection_label")

            if "attack_label" in snapshot_data.columns:
                columns_to_drop.append("attack_label")

            snapshot_features = snapshot_data.drop(columns=columns_to_drop, axis=1).to_numpy()
        else:
    
            snapshot_features = snapshot_data.drop(columns=columns_to_drop, axis=1).to_numpy()

        if graph_type == "TDG" or graph_type == "ALL":
            print('Start Create TDG Graph')
            create_graph(out_path=out_path,
                        capture_name=capture_name,
                        graph_type="tdg_graph",
                        snapshot_data=snapshot_data,
                        snapshot_features=snapshot_features,
                        interval_number=interval_number,
                        stats=stats,
                        log_file=log_file,
                        full_benign=full_benign,
                        full_malicious=full_malicious,
                        mixed=mixed,
                        also_etdg=graph_type
                        )
        elif graph_type == "eTDG" or graph_type == "ALL":
            print('Start Create TDG and eTDG Graph')
            create_graph(out_path=out_path,
                        capture_name=capture_name,
                        graph_type="etdg_graph",
                        snapshot_data=snapshot_data,
                        snapshot_features=snapshot_features,
                        interval_number=interval_number,
                        stats=stats,
                        log_file=log_file,
                        full_benign=full_benign,
                        full_malicious=full_malicious,
                        mixed=mixed,
                        also_etdg=graph_type
                        )
        elif graph_type == "SIM"  or graph_type == "ALL":
            print('Start Create SIM Graph')
            create_graph(out_path=out_path,
                        capture_name=capture_name,
                        graph_type="sim_graph",
                        snapshot_data=snapshot_data,
                        snapshot_features=snapshot_features,
                        interval_number=interval_number,
                        stats=stats,
                        log_file=log_file,
                        full_benign=full_benign,
                        full_malicious=full_malicious,
                        mixed=mixed,
                        also_etdg=graph_type
                        )

    log_file.close()
    return stats


def read_single_file(data_path, out_path, graph_type, snapshot_interval, k, result_rules_path):
    """

    Args:
        result_rules_path (str):
        out_path (str):
        log_file (str):
        graph_type (str):
        snapshot_interval (int):
        k (int):
        result_rules_path (str):

    Returns:
        dict: dict with statistics for a given capture
    """
    capture_name = result_rules_path.split('/')[-1].split('.')[0]
    print(f"\nConsidering capture {capture_name}")

    stats = dict()
    # check if capture has been already considered

    if stats.get(capture_name) is None:
        stats[capture_name] = dict()

    if graph_type == "eTDG" or graph_type == "ALL":
        os.makedirs(os.path.join(out_path, capture_name,
                    "etdg_graph", "full_benign"), exist_ok=True)
        os.makedirs(os.path.join(out_path, capture_name,
                    "etdg_graph", "mixed"), exist_ok=True)
        os.makedirs(os.path.join(out_path, capture_name,
                    "etdg_graph", "full_malicious"), exist_ok=True)

        if stats[capture_name].get("etdg_graph") is None:
            stats[capture_name]["etdg_graph"] = dict()

    if graph_type == "TDG" or graph_type == "ALL":
        os.makedirs(os.path.join(out_path, capture_name,
                    "tdg_graph", "full_benign"), exist_ok=True)
        os.makedirs(os.path.join(out_path, capture_name,
                    "tdg_graph", "mixed"), exist_ok=True)
        os.makedirs(os.path.join(out_path, capture_name,
                    "tdg_graph", "full_malicious"), exist_ok=True)

        if stats[capture_name].get("tdg_graph") is None:
            stats[capture_name]["tdg_graph"] = dict()
    
    if graph_type == "SIM" or graph_type == "ALL":
        os.makedirs(os.path.join(out_path, capture_name,
                    "sim_graph", "full_benign"), exist_ok=True)
        os.makedirs(os.path.join(out_path, capture_name,
                    "sim_graph", "mixed"), exist_ok=True)
        os.makedirs(os.path.join(out_path, capture_name,
                    "sim_graph", "full_malicious"), exist_ok=True)

        if stats[capture_name].get("sim_graph") is None:
            stats[capture_name]["sim_graph"] = dict()


    # create log file
    log_file_path = os.path.join(
            out_path, capture_name, f"{capture_name}.txt")

    # Read the file and sort by timestamp
    features = pd.read_csv(result_rules_path)
    # with open(os.path.join(
    #         data_path, f"{capture_name}.txt")) as f:
    #     lines = f.readlines()
    # # Parte di ordinamento e di divisione in chunk se il csv è grande
    # chunksize = 10000
    # total_flows = int(lines[-1].split(": ")[-1])/chunksize
    # features = pd.concat([chunk for chunk in tqdm(
    #     pd.read_csv(result_rules_path, chunksize=chunksize), total=total_flows, desc='Loading data')])
    features.sort_values(
        "bidirectional_first_seen_ms",  inplace=True)   
    
    
    #features.to_csv('mirai_ackflooding.csv', index=False)
    #print('Stampa ok')
    # compute number of snapshots
    # get the first and last timestamp
    start_ts = features.iloc[0]['bidirectional_first_seen_ms']
    end_ts = features.iloc[-1]['bidirectional_first_seen_ms']

    # compute the number of snaphshots in file
    if start_ts < end_ts:
        total_snapshots = ceil((end_ts-start_ts)/snapshot_interval)
    print(f"Number of snapshots {total_snapshots}")        

    # end_ts = start_ts+snapshot_interval
    # for interval_number in tqdm(range(total_snapshots), desc="Snapshots"):
    # prerare arguments for parallelized function
    # compute the list of snapshots start and end ts
    time_intervals_list = []
    start_snap_ts = start_ts
    for _ in range(total_snapshots):
        end_snap_ts = start_snap_ts+snapshot_interval
        time_intervals_list.append((start_snap_ts, end_snap_ts))
        start_snap_ts = end_snap_ts

    # out_path, log_file, capture_name, k, features, graph_type, time_indx
    f = functools.partial(generate_graph_from_snapshot,
                          out_path,
                          log_file_path,
                          capture_name,
                          k,
                          features,
                          graph_type
                          )

    time_intervals = [[time_intervals_list[indx], indx]
                      for indx in range(total_snapshots)]

    with Pool(1) as p:
        stats_snapshots_list = p.map(f, time_intervals)

    for snapshot in stats_snapshots_list:
        for snapshot_representation in snapshot.keys():
            if stats[capture_name].get(snapshot_representation) is None:
                stats[capture_name][snapshot_representation] = dict()
            for graph_name in snapshot[snapshot_representation].keys():
                stats[capture_name][snapshot_representation][graph_name] = snapshot[snapshot_representation][graph_name]

    stasts_file = os.path.join(out_path, capture_name, "stats.json")
    with open(stasts_file, "w") as stats_writer:
        json.dump(stats, stats_writer)

    #log_file.close()

    return stats


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc


if __name__ == "__main__":

    import debugpy
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data_path', type=str, 
                        help="Path to folder containing csv features")
    parser.add_argument('--logs_path', type=str, 
                        help="Path where log files will be saved")
    parser.add_argument('--out_path', type=str,
                        help="Path where graph will be saved")
    parser.add_argument('--graph_repre', type=str, 
                        help="Which kind of graphs to create [TDG, SIM, ALL]")
    parser.add_argument('--dataset', type=str, 
                        help="Dataset name" )
    parser.add_argument('--fast_mode', type=bool, 
                        default = True,
                        help = "Speeding up the execution process of iotid20 for eTDG-ALL")
    parser.add_argument('--intersection_iot23', type=bool, 
                        default = True,
                        help = "To be used when first eTDG and then you want the intersection with TDG")
    parser.add_argument('--snapshot_interval', type=int,
                        help="Snapshot time in ms")
    parser.add_argument('--compute_stats', action='store_true')

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(('0.0.0.0', 5679))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    data_path = args.data_path
    out_path = args.out_path
    logs_path = args.logs_path
    snapshot_interval = args.snapshot_interval
    graph_type = args.graph_repre
    dataset = args.dataset
    fast_mode = args.fast_mode
    intersect = args.intersection_iot23

    if "perc" in out_path:
        graph_type = "eTDG"
    print(graph_type)
    k = 20

    if args.dataset == 'IoT23':
        files_to_remove =  [f"{data_path}/CTU-IoT-Malware-Capture-17-1.csv",
                            f"{data_path}/CTU-IoT-Malware-Capture-33-1.csv", 
                            f"{data_path}/CTU-IoT-Malware-Capture-39-1.csv",
                            f"{data_path}/CTU-IoT-Malware-Capture-52-1.csv"]
        
        if graph_type == 'eTDG' or graph_type == 'ALL':
            files_to_remove += [f"{data_path}/CTU-IoT-Malware-Capture-7-1.csv",
                            f"{data_path}/CTU-IoT-Malware-Capture-9-1.csv", 
                            f"{data_path}/CTU-IoT-Malware-Capture-35-1.csv",
                            f"{data_path}/CTU-IoT-Malware-Capture-36-1.csv",
                            f"{data_path}/CTU-IoT-Malware-Capture-43-1.csv",
                            f"{data_path}/CTU-IoT-Malware-Capture-49-1.csv", 
                            f"{data_path}/CTU-IoT-Malware-Capture-60-1.csv"]

        elif graph_type == 'TDG' and snapshot_interval != 10000 and not intersect:
            files_to_remove += [f"{data_path}/CTU-IoT-Malware-Capture-43-1.csv"]

        #Questo solo quando si fa prima ALL in quanto per fare eTDG si deve fare TDG e ci fa risparmiare tempo
        elif graph_type == 'TDG' and snapshot_interval != 10000 and intersect:
            files_to_remove += [f"{data_path}/CTU-Honeypot-Capture-4-1.csv",
                                f"{data_path}/CTU-Honeypot-Capture-5-1.csv",
                                f"{data_path}/CTU-Honeypot-Capture-7-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-1-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-3-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-7-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-8-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-20-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-21-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-34-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-42-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-43-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-44-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-48-1.csv"]

        elif graph_type == 'TDG' and snapshot_interval == 10000 and intersect:
            files_to_remove += [f"{data_path}/CTU-Honeypot-Capture-4-1.csv",
                                f"{data_path}/CTU-Honeypot-Capture-5-1.csv",
                                f"{data_path}/CTU-Honeypot-Capture-7-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-1-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-3-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-7-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-8-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-20-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-21-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-34-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-42-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-44-1.csv",
                                f"{data_path}/CTU-IoT-Malware-Capture-48-1.csv"]

        elif graph_type == 'eTDG' or graph_type == 'ALL' and snapshot_interval == 10000:
            print("Error it is not possible to extract this files with eTDG mode")
            exit(1)

        result_rules_paths = [f"{data_path}/CTU-Honeypot-Capture-4-1.csv",
                              f"{data_path}/CTU-Honeypot-Capture-5-1.csv",
                              f"{data_path}/CTU-Honeypot-Capture-7-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-1-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-3-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-7-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-8-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-9-1.csv", 
                              f"{data_path}/CTU-IoT-Malware-Capture-17-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-20-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-21-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-33-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-34-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-35-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-36-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-39-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-42-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-43-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-44-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-48-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-49-1.csv", 
                              f"{data_path}/CTU-IoT-Malware-Capture-52-1.csv",
                              f"{data_path}/CTU-IoT-Malware-Capture-60-1.csv"] 

        result_rules_paths = list(
            set(result_rules_paths) - set(files_to_remove))
    elif args.dataset == 'IoTID20':
        files_to_remove = []
        if fast_mode and (graph_type == 'eTDG' or graph_type == 'ALL'):
            #Inserire not fast_mode
            files_to_remove += [f"{data_path}/dos-synflooding-1-dec.csv",
                                f"{data_path}/dos-synflooding-2-dec.csv",
                                f"{data_path}/dos-synflooding-3-dec.csv"]
        result_rules_paths = glob.glob(
            f"{data_path}/*.csv")
        result_rules_paths = list(
            set(result_rules_paths) - set(files_to_remove))
    elif args.dataset == 'IoT_Traces':
        files_to_remove = []
        result_rules_paths = glob.glob(
            f"{data_path}/*.csv")
        result_rules_paths = list(
            set(result_rules_paths) - set(files_to_remove))
    ## NEW CODE
    elif args.dataset == 'IoT23_clean':
        files_to_remove = []
        result_rules_paths = glob.glob(
            f"{data_path}/*.csv")
        result_rules_paths = list(
            set(result_rules_paths) - set(files_to_remove))
    ## NEW CODE
    elif args.dataset == 'Bot-IoT':
        files_to_remove = []
        result_rules_paths = glob.glob(
            f"{data_path}/*.csv")
        result_rules_paths = list(
            set(result_rules_paths) - set(files_to_remove))


    # Sort the file path based on file dimension
    dimension = []
    for file in result_rules_paths:
        dimension.append(os.path.getsize(file))
    order_indx = np.argsort(dimension)
    result_rules_paths_ordered = [result_rules_paths[i] for i in order_indx]
    
    print('File to manage')
    for i in result_rules_paths:
        print(i) 
    
    #
    # dict of statistichs
    stats = dict()
    cnt_dict = dict()

    if not args.compute_stats:
        # create log file
        os.makedirs(out_path, exist_ok=True)
        # data_path, out_path, graph_type, snapshot_interval, k, result_rules_path,
        f = functools.partial(read_single_file,
                              data_path,
                              out_path,
                              graph_type,
                              snapshot_interval,
                              k
                              )
        # with NoDaemonProcessPool(1) as p:
        #     stats_list = p.map(f,
        #                        result_rules_paths_ordered)
        stats_list = [f(result_rules_paths_ordered[i])
                      for i in range(len(result_rules_paths_ordered))]

        for capture in stats_list:
            for capture_name in capture.keys():
                stats[capture_name] = capture[capture_name]
        # write stats
        """
        stasts_file = os.path.join(out_path, "stats.json")
        with open(stasts_file, "w") as stats_writer:
            json.dump(stats, stats_writer)
        """
    else:
        for result_rules_path in result_rules_paths_ordered:
            capture_name = result_rules_path.split('/')[-1].split('.')[0]
            print(f"Considering capture {capture_name}")
            stats[capture_name] = dict()
            # if "IoT" in capture_name:
            if os.path.isdir(os.path.join(out_path, capture_name)):

                for graph_type in os.listdir(os.path.join(out_path, capture_name)):
                    if graph_type != "etdg_graph":
                        if os.path.isdir(os.path.join(out_path, capture_name, graph_type)):
                            stats[capture_name][graph_type] = dict()
                            if cnt_dict.get(graph_type) is None:
                                cnt_dict[graph_type] = dict()

                            for class_graph in os.listdir(os.path.join(out_path, capture_name, graph_type)):
                                if os.path.isdir(os.path.join(out_path, capture_name, graph_type, class_graph)):
                                    stats[capture_name][graph_type][class_graph] = 0
                                    if cnt_dict[graph_type].get(class_graph) is None:
                                        cnt_dict[graph_type][class_graph] = 0

                                    for file in os.listdir(os.path.join(out_path, capture_name, graph_type, class_graph)):
                                        if "full_graph" not in file:
                                            stats[capture_name][graph_type][class_graph] += 1
                                            cnt_dict[graph_type][class_graph] += 1

        print(f"Counters {cnt_dict}")
        """
        with open(f"{dataset}_dataset_stats.json", "w") as outfile:
            # json_data refers to the above JSON
            json.dump(stats, outfile)
        """