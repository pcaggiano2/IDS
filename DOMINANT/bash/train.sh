#!/bin/bash

# Set the CUDA device to use
# export CUDA_VISIBLE_DEVICES=3

# Define the base paths and constant values
GRAPHS_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/Graphs"
SPLITS_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit"
BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/DOMINANT/model"
CHECKPOINTS_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/Checkpoints/DOMINANT"
CSV_RESULT_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/Checkpoints/DOMINANT"

# Define the configurations
configurations=(
    # Each configuration contains: DATA_FOLD, MODEL, JSON_FOLD, GRAPH_TYPE, NORMALIZE e MIN_MAX_FOLDER

    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_tdg IoT23/60000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23 0"

    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoT23/60000/base/IoT23_dataset_split_sim sim_graph 1 IoT23 0"

    # "IoT23/90000/base DOMINANT_B4_64_90k_NORM_IoT23_tdg IoT23/90000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23 0"

    # "IoT23/90000/base DOMINANT_B4_64_90k_NORM_IoT23_sim IoT23/90000/base/IoT23_dataset_split_sim sim_graph 1 IoT23 0"
    
    # "IoT23/120000/base DOMINANT_B4_64_120k_NORM_IoT23_tdg IoT23/120000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23 0"
    
    # "IoT23/120000/base DOMINANT_B4_64_120k_NORM_IoT23_sim IoT23/120000/base/IoT23_dataset_split_sim sim_graph 1 IoT23 0"

    # "IoT23/150000/base DOMINANT_B4_64_150k_NORM_IoT23_tdg IoT23/150000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23 0"

    # "IoT23/150000/base DOMINANT_B4_64_150k_NORM_IoT23_sim IoT23/150000/base/IoT23_dataset_split_sim sim_graph 1 IoT23 0"

    # 5 Folds - Cross Validation - On Best  DOMINANT_B4_64_60k_NORM_IoT23_sim
    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoT23/60000/base/5folds sim_graph 1 IoT23 1"
    # fold 2 gia fatto
    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoT23/60000/base/5folds sim_graph 1 IoT23 3"
    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoT23/60000/base/5folds sim_graph 1 IoT23 4"
    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoT23/60000/base/5folds sim_graph 1 IoT23 5"


)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r DATA_FOLD MODEL JSON_FOLD GRAPH_TYPE NORM MINMAX FOLD_IDX <<< "$config"

    DATASET_FOLDER="$GRAPHS_PATH/$DATA_FOLD"
    JSON_FOLDER="$SPLITS_PATH/$JSON_FOLD"
    CHECKPOINT_FOLDER="$CHECKPOINTS_BASE_PATH/$MODEL/fold$FOLD_IDX/checkpoints"
    CSV_RESULT_PATH="$CSV_RESULT_BASE_PATH/$MODEL/fold$FOLD_IDX/train_val_results"
    MINMAX_PATH="$SPLITS_PATH/$MINMAX"

    echo "Processing configuration:"    
    echo "$MODEL"
    echo "$DATASET_FOLDER"
    echo "$JSON_FOLDER"
    echo "$GRAPH_TYPE"
    echo "$NORM"
    echo "$MINMAX_PATH"
    echo "$FOLD_IDX"

    # Ensure the output and checkpoint directories exist
    mkdir -p "$CHECKPOINT_FOLDER" "$CSV_RESULT_PATH"

    # Run the Python script with the current configuration
    python train.py \
        --dataset_folder "$DATASET_FOLDER" \
        --json_folder "$JSON_FOLDER" \
        --graph_type "$GRAPH_TYPE" \
        --checkpoint_folder "$CHECKPOINT_FOLDER" \
        --csv_results_folder "$CSV_RESULT_PATH" \
        --model "$MODEL" \
        --min_max "$MINMAX_PATH" \
        --normalize "$NORM" \
        --fold_idx "$FOLD_IDX"

done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project/
