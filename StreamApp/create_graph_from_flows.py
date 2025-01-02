import numpy as np

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


def create_graph(graph_type="sim_graph", snapshot_data=None, snapshot_features=None, interval_number=None):

    if graph_type == "sim_graph":
        # node_features_matrix, adj_matrix, labels_matrix, (node_array1, node_array2), node_reverse_dict, edges_list, node_ids_dict
        features_matrix, adj_matrix, labels_matrix, edge_list  = create_similarity_graph(
        snapshot_data=snapshot_data,
        snapshot_features=snapshot_features)
        graph = {"node_features": features_matrix, "adj": adj_matrix,
                "node_labels": labels_matrix, "edge_list": edge_list}
    else:
        features_matrix, adj_matrix, labels_matrix, edge_list, node_id_to_port_ip, edges_list, node_ids_dict, flow_label_matrix = create_tdg(
            snapshot_mat=snapshot_data.to_numpy(),
            snapshot_features=snapshot_features,
            label=False)
        graph = {"node_features": features_matrix, "adj": adj_matrix,
                 "node_labels": labels_matrix, "edge_list": edges_list,
                 "node_id_to_port_ip": node_id_to_port_ip,
                 "flow_label_matrix": flow_label_matrix}
       
    return graph
 