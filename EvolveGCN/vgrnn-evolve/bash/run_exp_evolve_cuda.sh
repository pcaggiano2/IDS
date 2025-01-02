#!/bin/bash
ulimit -n 65536
export CUDA_VISIBLE_DEVICES=0

BASE_PATH="/user/pcaggiano/IDS/EvolveGCN/vgrnn-evolve"
EXP_BASE_PATH="/user/pcaggiano/IDS/EvolveGCN/vgrnn-evolve/experiments"

configurations=(
    # "EvolveGCN_60k_NORM_IoT23_tdg/parameters_egcn_h_cuda.yaml"
    # "EvolveGCN_90k_NORM_IoT23_tdg/parameters_egcn_h_cuda.yaml"
    # "EvolveGCN_120k_NORM_IoT23_tdg/parameters_egcn_h_cuda.yaml"
    "EvolveGCN_60k_NORM_IoT23_tdg/parameters_egcn_o_cuda.yaml" # DA CAPIRE
    # "EvolveGCN_90k_NORM_IoT23_tdg/parameters_egcn_o_cuda.yaml"
    # "EvolveGCN_120k_NORM_IoT23_tdg/parameters_egcn_o_cuda.yaml"
)

cd "$BASE_PATH"
# Iterate over configurations

for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r DATA_PATH <<< "$config"
    

    echo "$DATA_PATH"
    YAML_FILE="$EXP_BASE_PATH/$DATA_PATH"
    python3 run_exp_anomaly.py --config_file "$YAML_FILE"

done

cd /user/pcaggiano/