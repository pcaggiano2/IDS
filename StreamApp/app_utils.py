import csv
from nfstream import NFStreamer
from math import ceil
from model import DOMINANT
from create_graph_from_flows import create_graph
import torch
from model_utils import *
from functional import *
import os



def from_protocol_number_to_protocol_code(protocol_dict, protocol_number):
    if protocol_number=='58':
        return 'icmp'
    return protocol_dict[protocol_number]

def load_protocol_dict(filename):
    protocol_dict={}
    dict_file = open(os.path.join(os.path.dirname(__file__), filename),"r")
    reader = csv.reader(dict_file, delimiter=",")
    for row in reader:
        protocol_dict[row[0]]=row[1].lower()
    for i in range(145,253):
        protocol_dict[str(i)]="unassigned"
    protocol_dict["253"]="test&implemnetation"
    protocol_dict["254"]="test&implemnetation"
    protocol_dict["255"]="reserved"
    return protocol_dict 

### CORE FUNCTIONS

def flow_extraction(uploaded_file):
    flow_streamer = NFStreamer(source=uploaded_file,n_dissections=0, statistical_analysis=True)
    flow_df = flow_streamer.to_pandas(columns_to_anonymize=[])
    return flow_df

def graph_creation(flow_df):
    snap_interval=60000
    flow_df.sort_values("bidirectional_first_seen_ms", inplace=True) 
    start_ts = flow_df.iloc[0]['bidirectional_first_seen_ms']
    end_ts = flow_df.iloc[-1]['bidirectional_first_seen_ms']
    start_time = start_ts
    end_time =  start_ts+snap_interval
    total_snapshots = ceil((end_ts-start_ts)/snap_interval)
    print(f"Total snapshots: {total_snapshots}")

    graphs = []
    
    for interval_number in range(total_snapshots):
        snapshot_data = flow_df.loc[(flow_df["bidirectional_first_seen_ms"]>=start_time) & (flow_df["bidirectional_first_seen_ms"]<end_time)]
        number_flows = snapshot_data.shape[0]
        print(f"Number of flows in {number_flows}")
        if snapshot_data.shape[0]>0:
            # remove categorical features
            snapshot_features = snapshot_data.drop(columns=["id","expiration_id","src_ip","src_mac","src_oui","src_port","dst_ip","dst_mac","dst_oui",
            "dst_port","protocol","ip_version","vlan_id","tunnel_id","bidirectional_first_seen_ms","bidirectional_last_seen_ms", 
            "src2dst_first_seen_ms","src2dst_last_seen_ms","dst2src_first_seen_ms","dst2src_last_seen_ms"], axis=1).to_numpy()
      
            graph = create_graph("sim_graph", snapshot_data, snapshot_features, interval_number)
            # graph = create_graph("tdg_graph", snapshot_data, snapshot_features, interval_number)

            graphs.append(graph)
        start_time = start_time + snap_interval
        end_time = end_time + snap_interval
    
    return flow_df, graphs

def get_model(filename):
    in_dim = 57
    hidden_dim = 64
    encoder_layers = 3
    decoder_layers = 2
    dropout = 0.3
    model = DOMINANT(in_dim=in_dim,
                         hid_dim=hidden_dim,
                         encoder_layers=encoder_layers,
                         decoder_layers=decoder_layers,
                         dropout=dropout)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), filename), map_location=torch.device('cpu')))
    return model

def import_optimal_threshold(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as file:
        threshold = float(file.readline().strip())
    return float(threshold)    
     
def graph_predict(model, threshold, attrs, adj, adj_label, alpha):
    anomaly_score = calculate_anomaly_score(model, attrs, adj, adj_label, alpha)
    return (anomaly_score>threshold).astype(int), anomaly_score


def model_prediction(graphs, device, model, threshold):
    alpha = 0.8
    preds = [] 
    for graph in graphs:
        adj = graph['adj']
        feat = graph['node_features']
        loop_adj = adj + sp.eye(adj.shape[0])
        adj_norm = normalize_adj(loop_adj).toarray()
        min, max = find_min_max_range(os.path.join(os.path.dirname(__file__), "min_max"))
        node_features = torch.FloatTensor(normalize_attrs(
                        graph['node_features'], min, max))
        adj_matrix = torch.FloatTensor(adj_norm)
        edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
        labels = torch.FloatTensor(graph['node_labels'].flatten())
        adj, attrs, adj_label = edge_index, node_features, labels,
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs = attrs.to(device)
        pred, _ = graph_predict(model, threshold, attrs, adj, adj_label, alpha)
        preds.append(pred)
    return preds