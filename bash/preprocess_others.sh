#!/bin/bash

# Define the base paths and constant values
BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/dataset"
INPUT_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/OriginalDatasets"
OUTPUT_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/PreprocessedDatasets"

# Define the configurations with each configuration containing: Dataset, Interval, Out_Path
configurations=(
    # "IoT23/single_capture.csv IoT23/single_capture_modified.csv"
    # "IoTID20/single_capture.csv IoTID20/single_capture_modified.csv"
)

# Change to base directory
cd "$BASE_PATH" || { echo "BASE_PATH directory not found!"; exit 1; }

# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r INPUT_PATH OUTPUT_PATH <<< "$config"

    FULL_INPUT_PATH="$INPUT_BASE_PATH"/"$INPUT_PATH"
    FULL_OUTPUT_PATH="$OUTPUT_BASE_PATH"/"$OUTPUT_PATH"

    # Display the configuration being processed
    echo "Processing configuration:"
    echo "$FULL_INPUT_PATH"
    echo "$FULL_OUTPUT_PATH"

    # Run the Python script with the current configuration
    python preprocess_csv_others.py \
        --input_file "$FULL_INPUT_PATH" \
        --output_file "$FULL_OUTPUT_PATH" \

done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project/
