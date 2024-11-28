#!/bin/bash

# Define the base paths and constant values
BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/dataset"

# Define the configurations with each configuration containing: Dataset, Interval, Out_Path
configurations=(
    #IoT_traces
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT_traces/60000/base/IoT_traces_dataset_split_tdg/ test_benign.json"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT_traces/60000/base/IoT_traces_dataset_split_sim/ test_benign.json"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT_traces/90000/base/IoT_traces_dataset_split_tdg/ test_benign.json"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT_traces/90000/base/IoT_traces_dataset_split_sim/ test_benign.json"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT_traces/120000/base/IoT_traces_dataset_split_tdg/ test_benign.json"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT_traces/120000/base/IoT_traces_dataset_split_sim/ test_benign.json"

    # Bot-IoT
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/60000/base/Bot-IoT_dataset_split_tdg/ test_benign.json"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/60000/base/Bot-IoT_dataset_split_tdg/ test_malicious.json"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/60000/base/Bot-IoT_dataset_split_tdg/ test_mixed.json "

    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/60000/base/Bot-IoT_dataset_split_sim/ test_benign.json"
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/60000/base/Bot-IoT_dataset_split_sim/ test_malicious.json"  
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/60000/base/Bot-IoT_dataset_split_sim/ test_mixed.json"
    
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/90000/base/Bot-IoT_dataset_split_tdg/ test_benign.json"
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/90000/base/Bot-IoT_dataset_split_tdg/ test_malicious.json"  
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/90000/base/Bot-IoT_dataset_split_tdg/ test_mixed.json"

    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/90000/base/Bot-IoT_dataset_split_sim/ test_benign.json"
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/90000/base/Bot-IoT_dataset_split_sim/ test_malicious.json"  
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/90000/base/Bot-IoT_dataset_split_sim/ test_mixed.json"

    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/120000/base/Bot-IoT_dataset_split_tdg/ test_benign.json"
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/120000/base/Bot-IoT_dataset_split_tdg/ test_malicious.json"  
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/120000/base/Bot-IoT_dataset_split_tdg/ test_mixed.json"

    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/120000/base/Bot-IoT_dataset_split_sim/ test_benign.json"
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/120000/base/Bot-IoT_dataset_split_sim/ test_malicious.json"  
    "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/Bot-IoT/120000/base/Bot-IoT_dataset_split_sim/ test_mixed.json"
)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r SPLIT_PATH SPLIT_NAME  <<< "$config"

    # Display the configuration being processed
    echo "Processing configuration:"
    echo "$SPLIT_PATH"
    echo "$SPLIT_NAME"


    # Run the Python script with the current configuration
    python preprocess_iot_traces_splits.py \
        --split_path "$SPLIT_PATH" \
        --split_name "$SPLIT_NAME"
       
done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project/
