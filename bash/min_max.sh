#!/bin/bash

# IoT23 Dataset

BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/dataset"
DATA_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/IoT23"
SPLITS_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23"

# Define configurations with each entry containing: CONF_TYPE_DATA, train_val_split, temporal_split
configurations=(
    # Uncomment the following lines for IoT configurations 
    # It's the same for all IoT configuration because train graphs are the same
    "60000/base 60000/base/IoT23_dataset_split_tdg/train.json tdg_graph"
)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration
for config in "${configurations[@]}"; do
    # Read each configuration into variables
    read -r DATA JSON_PATH REPRE <<< "$config"

    # Set the paths for data and output
    DATA_PATH="$DATA_BASE_PATH/$DATA"
    OUT_PATH="$SPLITS_BASE_PATH"
    JSON_PATH="$SPLITS_BASE_PATH/$JSON_PATH"
    REPRE="$REPRE"
    
    echo "Processing configuration:"
    echo "  Data Path: $DATA_PATH"
    echo "  Output Path: $OUT_PATH"
    echo "  JSON Path: $JSON_PATH"
    echo "  Representation: $REPRE"


    # Execute the Python script with the configuration parameters
    python compute_min_max.py \
        --dataset_path "$DATA_PATH" \
        --output_folder "$OUT_PATH" \
        --json_path "$JSON_PATH" \
        --representation "$REPRE"

done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project