#!/bin/bash

# Set the CUDA device to use
# export CUDA_VISIBLE_DEVICES=3

# Define the base paths and constant values
GRAPHS_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/Graphs"
SPLITS_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit"
BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/DOMINANT/model"
CHECKPOINTS_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/Checkpoints/DOMINANT"

configurations=(
    # Each configuration contains: DATA_FOLD, MODEL, JSON_FOLD, GRAPH_TYPE

    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_tdg IoT23/60000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23"

    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoT23/60000/base/IoT23_dataset_split_sim sim_graph 1 IoT23"

    # "IoT23/90000/base DOMINANT_B4_64_90k_NORM_IoT23_tdg IoT23/90000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23"

    # "IoT23/90000/base DOMINANT_B4_64_90k_NORM_IoT23_sim IoT23/90000/base/IoT23_dataset_split_sim sim_graph 1 IoT23"

    # "IoT23/120000/base DOMINANT_B4_64_120k_NORM_IoT23_tdg IoT23/120000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23"
    
    # "IoT23/120000/base DOMINANT_B4_64_120k_NORM_IoT23_sim IoT23/120000/base/IoT23_dataset_split_sim sim_graph 1 IoT23"
)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r DATA_FOLD MODEL JSON_FOLD GRAPH_TYPE NORM MINMAX <<< "$config"

    DATASET_FOLDER="$GRAPHS_PATH/$DATA_FOLD"
    JSON_FOLDER="$SPLITS_PATH/$JSON_FOLD"
    CHECKPOINT_FOLDER="$CHECKPOINTS_PATH/$MODEL/checkpoints"
    THRESHOLD_PATH="$CHECKPOINTS_PATH/$MODEL/thresholds"
    MINMAX_PATH="$SPLITS_PATH/$MINMAX"

    echo "Processing configuration:"    
    echo "$MODEL"
    echo "$DATASET_FOLDER"
    echo "$JSON_FOLDER"
    echo "$GRAPH_TYPE"
    echo "$THRESHOLD_PATH"

    # Ensure the output and checkpoint directories exist
    mkdir -p "$CHECKPOINT_FOLDER" "$CSV_RESULT_PATH"

    # Run the Python script with the current configuration
    python find_scores.py \
        --dataset_folder "$DATASET_FOLDER" \
        --json_folder "$JSON_FOLDER" \
        --graph_type "$GRAPH_TYPE" \
        --checkpoint_path "$CHECKPOINT_FOLDER" \
        --threshold_path "$THRESHOLD_PATH" \
        --normalize "$NORM" \
        --min_max "$MINMAX_PATH"

done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project/
