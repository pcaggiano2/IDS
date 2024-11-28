#!/bin/bash

# Define the base paths and constant values
BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/dataset"

# Define the configurations with each configuration containing: Dataset, Interval, Out_Path
configurations=(
    # IoT23
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/60000/base/IoT23_dataset_split_tdg/ test_benign.json single_capture_modified train_val 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/60000/base/IoT23_dataset_split_tdg/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/60000/base/IoT23_dataset_split_tdg/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/60000/base/IoT23_dataset_split_sim/ test_benign.json single_capture_modified train_val 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/60000/base/IoT23_dataset_split_sim/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/60000/base/IoT23_dataset_split_sim/ test_mixed.json single_capture_modified test 0.8"
    
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/90000/base/IoT23_dataset_split_tdg/ test_benign.json single_capture_modified train_val 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/90000/base/IoT23_dataset_split_tdg/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/90000/base/IoT23_dataset_split_tdg/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/90000/base/IoT23_dataset_split_sim/ test_benign.json single_capture_modified train_val 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/90000/base/IoT23_dataset_split_sim/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/90000/base/IoT23_dataset_split_sim/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/120000/base/IoT23_dataset_split_tdg/ test_benign.json single_capture_modified train_val 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/120000/base/IoT23_dataset_split_tdg/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/120000/base/IoT23_dataset_split_tdg/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/120000/base/IoT23_dataset_split_sim/ test_benign.json single_capture_modified train_val 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/120000/base/IoT23_dataset_split_sim/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoT23/120000/base/IoT23_dataset_split_sim/ test_mixed.json single_capture_modified test 0.8"

    #IoTID20
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/60000/base/IoTID20_dataset_split_tdg/ test_benign.json single_capture_modified test 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/60000/base/IoTID20_dataset_split_tdg/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/60000/base/IoTID20_dataset_split_tdg/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/60000/base/IoTID20_dataset_split_sim/ test_benign.json single_capture_modified test 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/60000/base/IoTID20_dataset_split_sim/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/60000/base/IoTID20_dataset_split_sim/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/90000/base/IoTID20_dataset_split_tdg/ test_benign.json single_capture_modified test 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/90000/base/IoTID20_dataset_split_tdg/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/90000/base/IoTID20_dataset_split_tdg/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/90000/base/IoTID20_dataset_split_sim/ test_benign.json single_capture_modified test 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/90000/base/IoTID20_dataset_split_sim/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/90000/base/IoTID20_dataset_split_sim/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/120000/base/IoTID20_dataset_split_tdg/ test_benign.json single_capture_modified test 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/120000/base/IoTID20_dataset_split_tdg/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/120000/base/IoTID20_dataset_split_tdg/ test_mixed.json single_capture_modified test 0.8"

    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/120000/base/IoTID20_dataset_split_sim/ test_benign.json single_capture_modified test 0.8"
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/120000/base/IoTID20_dataset_split_sim/ test_malicious.json single_capture_modified test 0.8"  
    # "/Users/pasqualecaggiano/Desktop/Master/Project/GraphsSplit/IoTID20/120000/base/IoTID20_dataset_split_sim/ test_mixed.json single_capture_modified test 0.8"

)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r SPLIT_PATH SPLIT_NAME CAPTURE_NAME SPLIT_TYPE TRAIN_VAL_SPLIT <<< "$config"

    # Display the configuration being processed
    echo "Processing configuration:"
    echo "$SPLIT_PATH"
    echo "$SPLIT_NAME"
    echo "$CAPTURE_NAME" 
    echo "$SPLIT_TYPE" 
    echo "$TRAIN_VAL_SPLIT"

    # Run the Python script with the current configuration
    python preprocess_split.py \
        --split_path "$SPLIT_PATH" \
        --split_name "$SPLIT_NAME" \
        --capture_name "$CAPTURE_NAME" \
        --split_type "$SPLIT_TYPE" \
        --train_val_split "$TRAIN_VAL_SPLIT"
done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project/
