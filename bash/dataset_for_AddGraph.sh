#!/bin/bash


BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/dataset"
DATA_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/Graphs"
# Define the configurations
configurations=(
    # Uncomment the following lines for IoT23 configurations
    "IoT23/60000/base/"
    "IoT23/90000/base/"
    "IoT23/120000/base/"
    "IoT23/150000/base/"

    # # Uncomment the following lines for IoTID20 configurations
    # "IoTID20/60000/base/"
    # "IoTID20/90000/base/"
    # "IoTID20/120000/base/"
    # "IoTID20/150000/base/"

    # Uncomment the following lines for IoT_Traces configurations
    # "IoT_traces/60000/base/"
    # "IoT_traces/90000/base/"
    # "IoT_traces/120000/base/"
    # "IoT_traces/150000/base/"
    
    # # Uncomment the following lines for Bot-IoT configurations
    # "Bot-IoT/60000/base/"
    # "Bot-IoT/90000/base/"
    # "Bot-IoT/120000/base/"
    #"Bot-IoT/150000/base/"
)

cd "$BASE_PATH"
# Iterate over configurations
pwd
# Iterate over each configuration string
for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r CONF_TYPE_DATA <<< "$config"

    DATA_PATH="$DATA_BASE_PATH/$CONF_TYPE_DATA"
    echo "Work on: $CONF_TYPE_DATA"

    python prepare_dataset_for_AddGraph.py --data_path "$DATA_PATH"
done

# Change back to the original directory for further processing
cd /Users/pasqualecaggiano/Desktop/Master/Project/
