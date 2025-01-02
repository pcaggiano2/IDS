#!/bin/bash
ulimit -n 65536
export CUDA_VISIBLE_DEVICES=0

BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/EvolveGCN/vgrnn-evolve"
EXP_BASE_PATH="/Users/pasqualecaggiano/Desktop/Master/Project/IDS/EvolveGCN/vgrnn-evolve/experiments"

configurations=(
    # "EvolveGCN_60k_NORM_IoT23_tdg/parameters_egcn_h.yaml"
    # "EvolveGCN_90k_NORM_IoT23_tdg/parameters_egcn_h.yaml"
    # "EvolveGCN_120k_NORM_IoT23_tdg/parameters_egcn_h.yaml"
    
    # "EvolveGCN_60k_NORM_IoT23_tdg/parameters_egcn_o.yaml"
    # "EvolveGCN_90k_NORM_IoT23_tdg/parameters_egcn_o.yaml"
    # "EvolveGCN_120k_NORM_IoT23_tdg/parameters_egcn_o.yaml"

    # 5 fold cross validation
    "EGCN_H_2min_cross_val/parameters_egcn_h_fold1.yaml"
    # fold2 già fatto
    "EGCN_H_2min_cross_val/parameters_egcn_h_fold3.yaml"
    "EGCN_H_2min_cross_val/parameters_egcn_h_fold4.yaml"
)

cd "$BASE_PATH"
# Iterate over configurations

for config in "${configurations[@]}"; do
    # Read the configuration string into variables
    read -r DATA_PATH <<< "$config"
    
    echo "$DATA_PATH"
    YAML_FILE="$EXP_BASE_PATH/$DATA_PATH"
    python run_exp_anomaly.py --config_file "$YAML_FILE"
   
done

cd /Users/pasqualecaggiano/Desktop/Master/Project