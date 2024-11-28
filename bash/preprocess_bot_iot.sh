#!/bin/bash

# Define the base paths and constant values
BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/dataset"
DATA_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/OriginalDatasets/Bot-IoT"
OUT_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/PreprocessedDatasets/Bot-IoT"

# Define the configurations with each configuration containing: Dataset, Interval, Out_Path
configurations=(
    # "Ground_Truth/Keylogging.csv PCAPs/Theft/Keylogging Keylogging.csv"
    # "Ground_Truth/Data_exfiltration.csv PCAPs/Theft/Data_Exfiltration Data_exfiltration.csv"
    # "Ground_Truth/Service_Scan.csv PCAPs/Scan/Service Service_Scan.csv"
    # "Ground_Truth/OS_Scan.csv PCAPs/Scan/OS OS_Scan.csv"
    # "Ground_Truth/DoS_HTTP.csv PCAPs/DoS/DoS_HTTP DoS_HTTP.csv"

    # "Ground_Truth/DoS_TCP.csv PCAPs/DoS/DoS_TCP/1 DoS_TCP_1.csv"
    # "Ground_Truth/DoS_TCP.csv PCAPs/DoS/DoS_TCP/2 DoS_TCP_2.csv"

    # "Ground_Truth/DoS_UDP.csv PCAPs/DoS/DoS_UDP/1 DoS_UDP_1.csv"
    # "Ground_Truth/DoS_UDP.csv PCAPs/DoS/DoS_UDP/2 DoS_UDP_2.csv"
    
    # "Ground_Truth/DDoS_TCP.csv PCAPs/DDoS/DDoS_TCP/1 DDoS_TCP_1.csv"
    # "Ground_Truth/DDoS_TCP.csv PCAPs/DDoS/DDoS_TCP/2 DDoS_TCP_2.csv"

    # "Ground_Truth/DDoS_UDP.csv PCAPs/DDoS/DDoS_UDP/1 DDoS_UDP_1.csv"
    # "Ground_Truth/DDoS_UDP.csv PCAPs/DDoS/DDoS_UDP/2 DDoS_UDP_2.csv"
)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r GT_PATH PCAP_PATH OUT_PATH <<< "$config"

    FULL_GT_PATH="$DATA_BASE_PATH"/"$GT_PATH"
    FULL_PCAP_PATH="$DATA_BASE_PATH"/"$PCAP_PATH"
    FULL_OUT_PATH="$OUT_BASE_PATH"/"$OUT_PATH"

    # Display the configuration being processed
    echo "Processing configuration:"
    echo "$FULL_GT_PATH"
    echo "$FULL_PCAP_PATH"
    echo "$FULL_OUT_PATH" 

    # Run the Python script with the current configuration
    python extract_and_label_flows_Bot_IoT.py \
        --gt_csv_file "$FULL_GT_PATH" \
        --pcap_file "$FULL_PCAP_PATH" \
        --out_csv_file "$FULL_OUT_PATH" \

done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project/
