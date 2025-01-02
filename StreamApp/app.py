import streamlit as st
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from app_utils import *
import torch
import time
import tempfile
import os
import datetime


# Preprocessing function
def preprocessing(uploaded_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())  # Write file content to the temporary file
        temp_file_path = temp_file.name  # Get the temporary file path

    start = time.time()
    flow_df = flow_extraction(temp_file_path)
    flow_extraction_time = time.time() - start

    start = time.time()
    flow_df, graphs = graph_creation(flow_df)
    graph_creation_time = time.time() - start

    return flow_df, graphs, flow_extraction_time, graph_creation_time

# Anomaly detection function
def anomaly_detection(graphs, device, model, threshold):
    start = time.time()
    preds = model_prediction(graphs, device, model, threshold)
    pred_time = time.time() - start
    return preds, pred_time

# Start analysis function
def start_analysis(uploaded_file, analysis_bar):
    device = torch.device('cpu') 
    model = get_model("model/DOMINANT_model_124_sim")
    # model = get_model("model/DOMINANT_model_107_tdg")
    model.eval()
    threshold = import_optimal_threshold("model/optimal_threshold.txt")

    print(uploaded_file)

    analysis_bar.progress(30, "Preprocessing operation...")
    flow_df, graphs, flow_extraction_time, graph_creation_time = preprocessing(uploaded_file)
    analysis_bar.progress(60, "Anomaly detection operation...")
    preds, pred_time = anomaly_detection(graphs, device, model, threshold)
    analysis_bar.progress(100, "Detection completed.")

    attack_detected = 0
    total_samples = 0
    for pred in preds:
        attack_detected += np.sum(pred)
        total_samples += pred.shape[0]
    
    if attack_detected > 0:
        st.markdown("### 🚨 **ATTACK DETECTED**")
        st.markdown(f"**Number of attacks detected:** {attack_detected} out of {total_samples} samples.")
        
        with st.expander("🔍 **Detailed Graph Analysis**"):
            for i, (graph, pred) in enumerate(zip(graphs, preds)):
                st.markdown(f"**Graph {i + 1}**")
                
                attack_flows = f"Attack Flows: **{int(np.sum(pred))}/{graph['node_features'].shape[0]}**"
                edges = f"Edges: **{int(np.sum(graph['adj']))}**"
                status = "Status: **Attack Detected**" if np.sum(pred) > 0 else "Status: **No Attack**"
                
                # Combine information into a single line for compactness
                st.write(f"{attack_flows} | {edges} | {status}")

                st.divider()
    else:
        st.markdown("### ✅ **NO ATTACK DETECTED**")
        st.markdown("All graphs were analyzed, and no anomalies were found.")
    
    return flow_df, preds

def show_anomalous_flows(preds, flow_df, protocol_dict):
    """
    Display anomalous network flows from multiple graphs in a table using Streamlit.

    Parameters:
        preds (list of numpy.array): List of prediction labels for each graph,
                                     where each array contains 1 for attack and 0 otherwise.
        flow_df (pd.DataFrame): DataFrame containing flow data.
        protocol_dict (dict): Dictionary mapping protocol numbers to protocol codes.
    """
    st.markdown("### 🔍 **Anomalous Flows and Nodes**")

    table_data = []
    start_idx = 0

    for graph_idx, pred in enumerate(preds):
        # Determine the end index for this graph's flows
        end_idx = start_idx + len(pred)

        # Get the flows corresponding to this graph
        graph_flows = flow_df.iloc[start_idx:end_idx]

        # Identify anomalous flows
        anomalous_indices = np.where(pred == 1)[0]
        for idx in anomalous_indices:
            flow = graph_flows.iloc[idx]

            timestamp_s = flow['bidirectional_first_seen_ms'] / 1000
            dt = datetime.datetime.fromtimestamp(timestamp_s)

            table_data.append({
                "Timestamp": dt,
                "Source IP": flow['src_ip'],
                "Source Port": flow['src_port'],
                "Destination IP": flow['dst_ip'],
                "Destination Port": flow['dst_port'],
                "Protocol": from_protocol_number_to_protocol_code(protocol_dict, str(flow['protocol']).upper())
            })

        # Update the start index for the next graph
        start_idx = end_idx

    # Check if there are any anomalous flows
    if not table_data:
        st.info("No anomalous flows detected.")
        return

    # Convert table data to DataFrame for display
    display_data = pd.DataFrame(table_data)

    # Display the table
    st.dataframe(display_data, use_container_width=True)


def main():

    st.set_page_config(page_title="NIDS", layout="centered", page_icon="🔍")

    protocol_dict = load_protocol_dict("protocols_number_code.csv")
    
    st.title("Graph Neural Network IDS")
    st.caption("GNN-based Anomaly Detection for Network Traffic")
    st.divider()
    st.logo(os.path.join(os.path.dirname(__file__), "images/logo.png"), size="large")

    # File upload section
    st.subheader("Upload a PCAP File")
    uploaded_file = st.file_uploader("Upload a PCAP file containing network traffic data", type="pcap")

    # Analysis section
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Start Analysis
        if st.button("Start Analysis"):
            st.subheader("Analysis Results")
            analysis_bar = st.progress(0, text="Operation in progress. Please wait.")
            try:
                flow_df, preds = start_analysis(uploaded_file, analysis_bar)

                st.success("Analysis completed successfully!")

                show_anomalous_flows(preds, flow_df, protocol_dict)

            except Exception as e:
                st.error(f"Error during preprocessing or inference: {e}")
        
        # Reset Button
        if st.button("Reset"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    else:
        st.info("Please upload a file to start the analysis.")

if __name__ == "__main__":
    main()
