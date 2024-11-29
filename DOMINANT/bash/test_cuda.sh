#!/bin/bash

# Set the CUDA device to use
export CUDA_VISIBLE_DEVICES=0

#export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Define the base paths and constant values
GRAPHS_PATH="/user/pcaggiano/Graphs"
SPLITS_PATH="/user/pcaggiano/GraphsSplit"
BASE_PATH="/user/pcaggiano/IDS/DOMINANT/model"
CHECKPOINTS_PATH="/user/pcaggiano/IDS/Checkpoints/DOMINANT"

configurations=(
    # Each configuration contains: DATA_FOLD, MODEL, JSON_FOLD, GRAPH_TYPE, NORMALIZE, MIN_MAX_FOLDER, DARTASET TO TEST
    
    ## Experiments [DOMINANT_B4_64_60k_NORM_IoT23_tdg]
    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_tdg IoT23/60000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23 IoT23"
    # "IoTID20/60000/base DOMINANT_B4_64_60k_NORM_IoT23_tdg IoTID20/60000/base/IoTID20_dataset_split_tdg tdg_graph 1 IoT23 IoTID20"
    # "IoT_traces/60000/base DOMINANT_B4_64_60k_NORM_IoT23_tdg IoT_traces/60000/base/IoT_traces_dataset_split_tdg tdg_graph 1 IoT23 IoT_traces"
    # "Bot-IoT/60000/base DOMINANT_B4_64_60k_NORM_IoT23_tdg Bot-IoT/60000/base/Bot-IoT_dataset_split_tdg tdg_graph 1 IoT23 Bot-IoT" ## DA FARE

    ## Experiments [DOMINANT_B4_64_60k_NORM_IoT23_sim]
    # "IoT23/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoT23/60000/base/IoT23_dataset_split_sim sim_graph 1 IoT23 IoT23"
    # "IoTID20/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoTID20/60000/base/IoTID20_dataset_split_sim sim_graph 1 IoT23 IoTID20"
    # "IoT_traces/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim IoT_traces/60000/base/IoT_traces_dataset_split_sim sim_graph 1 IoT23 IoT_traces"
    # "Bot-IoT/60000/base DOMINANT_B4_64_60k_NORM_IoT23_sim Bot-IoT/60000/base/Bot-IoT_dataset_split_sim sim_graph 1 IoT23 Bot-IoT" ## DA FARE

    ## Experiments [DOMINANT_B4_64_90k_NORM_IoT23_tdg]
    # "IoT23/90000/base DOMINANT_B4_64_90k_NORM_IoT23_tdg IoT23/90000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23 IoT23"
    # "IoTID20/90000/base DOMINANT_B4_64_90k_NORM_IoT23_tdg IoTID20/90000/base/IoTID20_dataset_split_tdg tdg_graph 1 IoT23 IoTID20"
    # "IoT_traces/90000/base DOMINANT_B4_64_90k_NORM_IoT23_tdg IoT_traces/90000/base/IoT_traces_dataset_split_tdg tdg_graph 1 IoT23 IoT_traces"
    # "Bot-IoT/90000/base DOMINANT_B4_64_90k_NORM_IoT23_tdg Bot-IoT/90000/base/Bot-IoT_dataset_split_tdg tdg_graph 1 IoT23 Bot-IoT" ## DA FARE

    ## Experiments [DOMINANT_B4_64_90k_NORM_IoT23_sim]
    # "IoT23/90000/base DOMINANT_B4_64_90k_NORM_IoT23_sim IoT23/90000/base/IoT23_dataset_split_sim sim_graph 1 IoT23 IoT23"
    # "IoTID20/90000/base DOMINANT_B4_64_90k_NORM_IoT23_sim IoTID20/90000/base/IoTID20_dataset_split_sim sim_graph 1 IoT23 IoTID20"
    "IoT_traces/90000/base DOMINANT_B4_64_90k_NORM_IoT23_sim IoT_traces/90000/base/IoT_traces_dataset_split_sim sim_graph 1 IoT23 IoT_traces"
    # "Bot-IoT/90000/base DOMINANT_B4_64_90k_NORM_IoT23_sim Bot-IoT/90000/base/Bot-IoT_dataset_split_sim sim_graph 1 IoT23 Bot-IoT" ## DA FARE 

    ## Experiments [DOMINANT_B4_64_120k_NORM_IoT23_tdg]
    # "IoT23/120000/base DOMINANT_B4_64_120k_NORM_IoT23_tdg IoT23/120000/base/IoT23_dataset_split_tdg tdg_graph 1 IoT23 IoT23"
    # "IoTID20/120000/base DOMINANT_B4_64_120k_NORM_IoT23_tdg IoTID20/120000/base/IoTID20_dataset_split_tdg tdg_graph 1 IoT23 IoTID20"
    "IoT_traces/120000/base DOMINANT_B4_64_120k_NORM_IoT23_tdg IoT_traces/120000/base/IoT_traces_dataset_split_tdg tdg_graph 1 IoT23 IoT_traces"
    # "Bot-IoT/120000/base DOMINANT_B4_64_120k_NORM_IoT23_tdg Bot-IoT/120000/base/Bot-IoT_dataset_split_tdg tdg_graph 1 IoT23 Bot-IoT" ## DA FARE

    ## Experiments [DOMINANT_B4_64_120k_NORM_IoT23_sim]
    # "IoT23/120000/base DOMINANT_B4_64_120k_NORM_IoT23_sim IoT23/120000/base/IoT23_dataset_split_sim sim_graph 1 IoT23 IoT23"
    # "IoTID20/120000/base DOMINANT_B4_64_120k_NORM_IoT23_sim IoTID20/120000/base/IoTID20_dataset_split_sim sim_graph 1 IoT23 IoTID20"
    "IoT_traces/120000/base DOMINANT_B4_64_120k_NORM_IoT23_sim IoT_traces/120000/base/IoT_traces_dataset_split_sim sim_graph 1 IoT23 IoT_traces"
    # "Bot-IoT/120000/base DOMINANT_B4_64_120k_NORM_IoT23_sim Bot-IoT/120000/base/Bot-IoT_dataset_split_sim sim_graph 1 IoT23 Bot-IoT" ## DA FARE 
)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r DATA_FOLD MODEL JSON_FOLD GRAPH_TYPE NORM MINMAX DATASET <<< "$config"

    DATASET_FOLDER="$GRAPHS_PATH/$DATA_FOLD"
    JSON_FOLDER="$SPLITS_PATH/$JSON_FOLD"
    CHECKPOINT_FOLDER="$CHECKPOINTS_PATH/$MODEL/checkpoints"
    RESULT_PATH="$CHECKPOINTS_PATH/$MODEL/$DATASET/y_pred_true"
    MINMAX_PATH="$SPLITS_PATH/$MINMAX"
    THRESHOLD_PATH="$CHECKPOINTS_PATH/$MODEL/thresholds"

    echo "Processing configuration:"    
    echo "$MODEL"
    echo "$DATASET_FOLDER"
    echo "$JSON_FOLDER"
    echo "$GRAPH_TYPE"
    echo "$THRESHOLD_PATH"

    # Ensure the output and checkpoint directories exist
    mkdir -p "$CHECKPOINT_FOLDER" "$CSV_RESULT_PATH"

   echo "Work on: $MODEL"
    python3 test.py --dataset_folder "${DATASET_FOLDER}" \
    --json_folder "${JSON_FOLDER}" \
    --graph_type "${GRAPH_TYPE}"\
    --checkpoint_path "${CHECKPOINT_FOLDER}"\
    --result_path "${RESULT_PATH}" \
    --dataset "${DATASET}" \
    --normalize "${NORM}" \
    --threshold_path "${THRESHOLD_PATH}" \
    --min_max "${MINMAX_PATH}" \
    --device "cuda"

done

# Change back to the original directory for further processing
cd /user/pcaggiano
