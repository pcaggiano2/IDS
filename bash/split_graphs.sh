#!/bin/bash

# Define the base path
BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/dataset"

# Define configurations with each entry containing: CONF_TYPE_DATA, train_val_split, temporal_split
configurations=(
    # Uncomment the following lines for IoT configurations
    # "IoT_traces/60000/base 0.8 0.8"
    # "IoT_traces/90000/base 0.8 0.8"
    # "IoT_traces/120000/base 0.8 0.8"
    # "IoT_traces/150000/base 0.8 0.8"

    # Uncomment the following lines for IoT23 configurations
    # "IoT23/60000/base 0.8 0.8"
    # "IoT23/90000/base 0.8 0.8"
    # "IoT23/120000/base 0.8 0.8"
    # "IoT23/150000/base 0.8 0.8"

    # Uncomment the following lines for IoTID20 configurations
    # "IoTID20/60000/base 0.8 0.8"
    # "IoTID20/90000/base 0.8 0.8"
    # "IoTID20/120000/base 0.8 0.8"
    # "IoTID20/150000/base 0.8 0.8"

    # Uncomment the following lines for IoTID20 configurations
    # "Bot-IoT/60000/base 0.8 0.8"
    # "Bot-IoT/90000/base 0.8 0.8"
    # "Bot-IoT/120000/base 0.8 0.8"
    "Bot-IoT/150000/base 0.8 0.8"
)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration
for config in "${configurations[@]}"; do
    # Read each configuration into variables
    read -r CONF_TYPE_DATA TRAIN_VAL_SPLIT TEMPORAL_SPLIT <<< "$config"

    # Set the paths for data and output
    DATA_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/Graphs/$CONF_TYPE_DATA"
    OUT_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/$CONF_TYPE_DATA"
    
    echo "Processing configuration:"
    echo "  Data Path: $DATA_PATH"
    echo "  Output Path: $OUT_PATH"
    echo "  Train-Validation Split: $TRAIN_VAL_SPLIT"
    echo "  Temporal Split: $TEMPORAL_SPLIT"

    Execute the Python script with the configuration parameters
    # python create_split.py \
    #     --data_path "$DATA_PATH" \
    #     --out_path "$OUT_PATH" \
    #     --train_val_split "$TRAIN_VAL_SPLIT" \
    #     --temporal_split "$TEMPORAL_SPLIT"
    
    # Uncomment below for only the test phase
    python create_split.py \
        --data_path "$DATA_PATH" \
        --out_path "$OUT_PATH" \
        --train_val_split "$TRAIN_VAL_SPLIT" \
        --temporal_split "$TEMPORAL_SPLIT" \
        --only_test

    # Check if the script was successful
    if [ $? -ne 0 ]; then
        echo "Error: Script failed for configuration CONF_TYPE_DATA=$CONF_TYPE_DATA"
        exit 1
    fi
done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project