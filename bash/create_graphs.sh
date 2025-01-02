#!/bin/bash

# Define the base paths and constant values
OUT_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/Graphs"
BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/dataset"
BASE_DATA_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/PreprocessedDatasets/"

# GRAPH_REPRE="TDG"           # Options: TDG, eTDG, SIM or ALL
GRAPH_REPRE="SIM"   
# Define the configurations with each configuration containing: Dataset, Interval, Out_Path
configurations=(
    # # Uncomment the following lines for IoT23 configurations
    # "IoT23_clean 60000 IoT23/60000/base/"
    # "IoT23_clean 90000 IoT23/90000/base/"
    # "IoT23_clean 120000 IoT23/120000/base/"
    # "IoT23_clean 150000 IoT23/150000/base/"

    # # # Uncomment the following lines for IoTID20 configurations
    # "IoTID20 60000 IoTID20/60000/base/"
    # "IoTID20 90000 IoTID20/90000/base/"
    # "IoTID20 150000 IoTID20/150000/base/"
    # "IoTID20 120000 IoTID20/120000/base/"

    # # Uncomment the following lines for IoT_Traces configurations
    # "IoT_Traces 60000 IoT_traces/60000/base/"
    # "IoT_Traces 90000 IoT_traces/90000/base/"
    # "IoT_Traces 120000 IoT_traces/120000/base/"
    # "IoT_Traces 150000 IoT_traces/150000/base/"
    
    # # Uncomment the following lines for Bot-IoT configurations
    # "Bot-IoT 60000 Bot-IoT/60000/base/"
    # "Bot-IoT 90000 Bot-IoT/90000/base/"
    # "Bot-IoT 120000 Bot-IoT/120000/base/"
    # "Bot-IoT 150000 Bot-IoT/150000/base/"
)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r DATASET INTERVAL OUT_FILE <<< "$config"
    
    # Define full paths for output and logs
    if [ "$DATASET" = "IoT23_clean" ]; then
        DATA_PATH="$BASE_DATA_PATH/IoT23"
    else
        DATA_PATH="$BASE_DATA_PATH/$DATASET"
    fi
    OUT_PATH="$OUT_BASE_PATH/$OUT_FILE"
    LOGS_PATH="$OUT_PATH/logs"

    # Ensure output and log directories exist
    mkdir -p "$OUT_PATH" "$LOGS_PATH"

    # Display the configuration being processed
    echo "Processing configuration:"
    echo "  Dataset: $DATASET"
    echo "  Interval: $INTERVAL"
    echo "  Output Path: $OUT_PATH"
    echo "  Logs Path: $LOGS_PATH"

    # Run the Python script with the current configuration
    python graph_representation_creation.py \
        --data_path "$DATA_PATH" \
        --logs_path "$LOGS_PATH" \
        --out_path "$OUT_PATH" \
        --graph_repre "$GRAPH_REPRE" \
        --snapshot_interval "$INTERVAL" \
        --dataset "$DATASET"
    
done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project/