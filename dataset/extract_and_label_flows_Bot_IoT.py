import pandas as pd
import numpy as np
import os
from nfstream import NFStreamer
import argparse
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# # Define the function to apply to each row, with ground_truth_df as a parameter
# def label_flow_parallel(row, ground_truth_df):
#     return label_flow(row, ground_truth_df)

# Function to check if two time windows overlap
def is_time_overlap(flow_start, flow_end, gt_start, gt_end):
    value = (flow_start <= gt_end) and (flow_end >= gt_start)
    return value

# Function to label a flow by matching against the ground truth
def label_flow(flow_row, ground_truth_df):
    # Filter ground truth entries to reduce comparisons

    filtered_gt = ground_truth_df[(ground_truth_df['stime'] == flow_row['bidirectional_first_seen_ms']) &
                                  (ground_truth_df['ltime'] == flow_row['bidirectional_last_seen_ms'])]
    
    # Iterate through the filtered ground truth entries to find a match
    for index, gt_row in filtered_gt.iterrows():
        # Check IP addresses, ports, and protocol
        # Check each condition separately with debug prints
        if flow_row['src_ip'] == gt_row['saddr']:
            if flow_row['dst_ip'] == gt_row['daddr']:
                if flow_row['src_port'] == gt_row['sport']:
                    if flow_row['dst_port'] == gt_row['dport']:
                        if flow_row['protocol'] == gt_row['proto']:
                            if is_time_overlap(flow_row['bidirectional_first_seen_ms'], 
                                            flow_row['bidirectional_last_seen_ms'], 
                                            gt_row['stime'], 
                                            gt_row['ltime']):
                                if gt_row['attack'] == 0:
                                    return 'Benign'
                                elif gt_row['attack'] == 1:
                                    return 'Malicious'
                                else:
                                    return 'Unknown'
                            else:
                                pass
                                # print("Time overlap check failed.")
                                # print(f"Flow start: {flow_row['bidirectional_first_seen_ms']}, GT start: {gt_row['stime']}")
                        else:
                            pass
                            # print("Protocol check failed.")
                            # print(f"Flow protocol: {flow_row['protocol']}, GT proto: {gt_row['proto']}")
                    else:
                        pass
                        # print("Destination port check failed.")
                        # print(f"Flow dst_port: {flow_row['dst_port']}, GT dport: {gt_row['dport']}")
                else:
                    pass
                    # print("Source port check failed.")
                    # print(f"Flow src_port: {flow_row['src_port']}, GT sport: {gt_row['sport']}")
            else:
                pass
                # print("Destination IP check failed.")
                # print(f"Flow dst_ip: {flow_row['dst_ip']}, GT daddr: {gt_row['daddr']}")
        else:
            pass
            # print("Source IP check failed.")
            # print(f"Flow src_ip: {flow_row['src_ip']}, GT saddr: {gt_row['saddr']}")

    return 'Unknown'  # Return 'Unknown' if no match is found

# Helper function to apply label_flow_parallel to each row in a DataFrame chunk
def apply_func_to_chunk(df_chunk, ground_truth_df):
    # Apply the function to each row, passing in ground_truth_df
    return df_chunk.apply(lambda row: label_flow(row, ground_truth_df), axis=1)

# Split DataFrame into chunks for parallel processing
def parallel_apply(df, ground_truth_df, num_partitions):
    # Split the DataFrame into smaller chunks
    df_split = np.array_split(df, num_partitions)
    
    # Use ProcessPoolExecutor to apply the function in parallel
    with ProcessPoolExecutor() as executor:
        # Map each chunk to apply_func_to_chunk with ground_truth_df as argument
        results = executor.map(partial(apply_func_to_chunk, ground_truth_df=ground_truth_df), df_split)
    
    # Concatenate the results back into a single DataFrame column
    return pd.concat(results)



def save_counts_json(out_csv_file, flow_df, ground_truth_df, file_path="label_counts.json"):
    # Count occurrences of each unique value in flow_df['detection_label']
    label_counts_flow = flow_df['detection_label'].value_counts()
    print("Counts of unique values in flow_df['detection_label']:")
    print(label_counts_flow)

    label_counts_flow_dict = label_counts_flow.to_dict()

    label_counts_ground_truth = ground_truth_df['attack'].value_counts()
    print("Counts of unique values in ground_truth_df['detection_label']:")
    print(label_counts_ground_truth)

    label_counts_ground_truth_dict = label_counts_ground_truth.to_dict()

    # Combine both counts into a single dictionary
    new_counts = {
        "out_csv_file": out_csv_file,
        "flow_df_detection_label_counts": label_counts_flow_dict,
        "ground_truth_df_attack_counts": label_counts_ground_truth_dict
    }

    # Check if the file exists
    if os.path.exists(file_path):
        # Load existing data
        with open(file_path, "r") as f:
            existing_data = json.load(f)
    else:
        # Initialize an empty list if file does not exist
        existing_data = []

    # Append the new counts to the existing data
    existing_data.append(new_counts)

    # Save the updated data back to the JSON file
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print("Counts appended to label_counts.json")

def debugging(ground_truth_df, flow_df):

    # Set pandas to display floats with 12 decimal places for better precision
    pd.set_option('display.float_format', lambda x: f'{x:.12f}')

    # Conut each row in the two dataframe

    print(f"Ground truth size: {ground_truth_df.shape[0]}")
    print(f"Flow size: {flow_df.shape[0]}")

    # # Print data types of 'sport' column in ground_truth_df and 'src_port' column in flow_df
    # print(ground_truth_df['sport'].dtypes)
    # print(flow_df['src_port'].dtypes)

    # # Count and print the number of missing (NaN) values in 'sport' and 'dport' columns in ground_truth_df
    # print(ground_truth_df['sport'].isna().sum())
    # print(ground_truth_df['dport'].isna().sum())

    # # Count and print the number of missing (NaN) values in 'src_port' and 'dst_port' columns in flow_df
    # print(flow_df['src_port'].isna().sum())
    # print(flow_df['dst_port'].isna().sum())

    # # Print the first 10 values of 'stime' column in ground_truth_df and 'bidirectional_first_seen_ms' column in flow_df
    # print(ground_truth_df['stime'].head(10))
    # print(flow_df['bidirectional_first_seen_ms'].head(10))
    # # Print the first 10 values of 'ltime' column in ground_truth_df and 'bidirectional_last_seen_ms' in flow_df
    # print(ground_truth_df['ltime'].head(10))
    # print(flow_df['bidirectional_last_seen_ms'].head(10))

    # # Print the shape of both DataFrames, which shows the number of rows and columns
    # print(ground_truth_df.shape)
    # print(flow_df.shape)

    # # Print the column names of both DataFrames to inspect their structure
    # print(ground_truth_df.columns)
    # print(flow_df.columns)

    # # Get unique values from the 'src_ip' column in flow_df and 'saddr' column in ground_truth_df
    # unique_src_ips = flow_df['src_ip'].unique()
    # unique_saddrs = ground_truth_df['saddr'].unique()

    # # Convert the unique values to sets for easier comparison
    # src_ip_set = set(unique_src_ips)
    # saddr_set = set(unique_saddrs)

    # # Find matching and non-matching IPs between the flow_df 'src_ip' and ground_truth_df 'saddr'
    # matching_ips = src_ip_set.intersection(saddr_set)
    # non_matching_src_ips = src_ip_set - matching_ips  # IPs in flow_df not in ground_truth_df
    # non_matching_saddrs = saddr_set - matching_ips  # Addresses in ground_truth_df not in flow_df

    # # Print the matching IP addresses
    # print("Matching Source IPs:")
    # for ip in matching_ips:
    #     print(ip)

    # # Print the non-matching source IPs from flow_df
    # print("\nNon-Matching Source IPs in flow_df:")
    # for ip in non_matching_src_ips:
    #     print(ip)

    # # Print the non-matching source addresses from ground_truth_df
    # print("\nNon-Matching Source Addresses in ground_truth_df:")
    # for addr in non_matching_saddrs:
    #     print(addr)

    # # Count and print the occurrences of a specific IP address ('192.168.100.1') in the 'saddr' column of ground_truth_df
    # ip_address = '192.168.100.1'
    # ip_count = (ground_truth_df['saddr'] == ip_address).sum()
    # print(f"Occurrences of IP address {ip_address}: {ip_count}")

    # # Count and print the occurrences of a specific MAC address ('00:00:00:00:00:00') in the 'smac' column of ground_truth_df
    # mac_address = '00:00:00:00:00:00'
    # mac_count = (ground_truth_df['smac'] == mac_address).sum()
    # print(f"Occurrences of MAC address {mac_address}: {mac_count}")

    # # For each matched IP address, print the number of occurrences in both the flow_df and ground_truth_df
    # for matched_ip in matching_ips:
    #     print(f"Occurrences in FLOW_DF of IP address {matched_ip}: {(ground_truth_df['saddr'] == matched_ip).sum()}")
    #     print(f"Occurrences in GT_DF of IP address {matched_ip}:{(flow_df['src_ip'] == matched_ip).sum()}")
    #     print("\n")

    # # Check and print the unique values in the 'proto' column of ground_truth_df
    # unique_proto_values = ground_truth_df['proto'].unique()
    # print("Unique values in ground_truth_df['proto']:", unique_proto_values)

    # # Check and print the unique values in the 'protocol' column of flow_df
    # unique_protocol_values = flow_df['protocol'].unique()
    # print("Unique values in flow_df['protocol']:", unique_protocol_values)

    # # Count and print occurrences of each unique value in the 'proto' column of ground_truth_df
    # proto_counts = ground_truth_df['proto'].value_counts()
    # print("Counts of unique values in ground_truth_df['proto']:")
    # print(proto_counts)

    # # Count and print occurrences of each unique value in the 'protocol' column of flow_df
    # protocol_counts = flow_df['protocol'].value_counts()
    # print("\nCounts of unique values in flow_df['protocol']:")
    # print(protocol_counts)

def read_groud_truth(gt_csv_file):
    ground_truth_df = pd.read_csv(gt_csv_file, sep=';')
    ground_truth_df['stime'] = ground_truth_df['stime'] * 1000
    ground_truth_df['ltime'] = ground_truth_df['ltime'] * 1000

    # Convert the 'stime' and 'ltime' columns to integer format
    ground_truth_df['stime'] = ground_truth_df['stime'].astype('int64')
    ground_truth_df['ltime'] = ground_truth_df['ltime'].astype('int64')

    # Convert the 'sport' and 'dport' columns to integer format
    ground_truth_df = ground_truth_df.dropna(subset=['sport', 'dport'])
    ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "0x0303") | (ground_truth_df['dport'] == "0x0303"))]
    ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "0x000c") | (ground_truth_df['dport'] == "0x000c"))]
    ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "0x0008") | (ground_truth_df['dport'] == "0x0008"))]
    ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "xinetd") | (ground_truth_df['dport'] == "xinetd"))]
    ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "nut") | (ground_truth_df['dport'] == "nut"))]
    ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "0x000d") | (ground_truth_df['dport'] == "0x000d"))]
    ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "0x0011") | (ground_truth_df['dport'] == "0x0011"))]
    ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "login") | (ground_truth_df['dport'] == "login"))]
    # ground_truth_df = ground_truth_df[~((ground_truth_df['sport'] == "0x000d") | (ground_truth_df['dport'] == "0x000d"))]


    ground_truth_df['sport'] = ground_truth_df['sport'].astype('int64')
    ground_truth_df['dport'] = ground_truth_df['dport'].astype('int64')

    # Replace 'tcp' with 6 and 'udp' with 17 in ground_truth_df['proto']
    ground_truth_df['proto'] = ground_truth_df['proto'].replace({'tcp': 6, 'udp': 17})

    return ground_truth_df

def extract_flow_from_folder(pcap_folder):
 
    dataframes = []

    for filename in os.listdir(pcap_folder):
        if filename.endswith(".pcap"):  
            pcap_file = os.path.join(pcap_folder, filename)
            
            streamer = NFStreamer(source=pcap_file, statistical_analysis=True, n_dissections=0)
            
            df = streamer.to_pandas(columns_to_anonymize=[])
            
            dataframes.append(df)

    complete_df = pd.concat(dataframes, ignore_index=True)

    return complete_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract and label flows from PCAP files.')
    parser.add_argument('--gt_csv_file', type=str, help='Path to the ground truth CSV file.')
    parser.add_argument('--pcap_file', type=str, help='Path to the PCAP file or folder containing PCAP files.')
    parser.add_argument('--out_csv_file', type=str, help='Path to the output CSV file.')

    args = parser.parse_args()

    gt_csv_file = args.gt_csv_file
    pcap_file = args.pcap_file
    out_csv_file = args.out_csv_file

    # Read Ground Truth file
    ground_truth_df = read_groud_truth(gt_csv_file)

    # Extract flow from PCAP files
    flow_df = extract_flow_from_folder(pcap_file)

    # Label Flows
    # flow_df['detection_label'] = flow_df.apply(lambda row: label_flow(row, ground_truth_df), axis=1)
    flow_df['detection_label'] = parallel_apply(flow_df, ground_truth_df, num_partitions=8)

    save_counts_json(out_csv_file,flow_df,ground_truth_df)

    # Remove rows where 'detection_label' is 'Unknown'
    flow_df = flow_df[flow_df['detection_label'] != 'Unknown']

    # Svae labeled flows as csv file
    flow_df.to_csv(out_csv_file, index=False)

    # debugging(ground_truth_df=ground_truth_df,flow_df=flow_df)


