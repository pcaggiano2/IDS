#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

BASE_PATH="/user/apaolillo/dynamic_graphs/EvolveGCN/"

configurations=(
    
    "150k_IoT23_etdg"
    #"120k_IoT23_etdg"
    #"90k_IoT23_etdg"
    #"60k_IoT23_etdg"
    
    #"150k_IoT23_tdg"
    #"120k_IoT23_tdg"
    #"90k_IoT23_tdg"
    #"60k_IoT23_tdg"
    
    #"150k_IoT_traces_etdg"
    #"120k_IoT_traces_etdg"
    #"90k_IoT_traces_etdg"
    #"60k_IoT_traces_etdg"
    
    #"150k_IoT_traces_tdg"
    #"120k_IoT_traces_tdg"
    #"90k_IoT_traces_tdg"
    #"60k_IoT_traces_tdg"
)

cd "$BASE_PATH"
# Iterate over configurations

for ((i=0; i<${#configurations[@]}; i+=1)); do
    DATA_PATH="${configurations[i]}"
    
    echo "EGCN-O norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_egcn_o_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"
    
    echo "EGCN-H norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_egcn_h_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"

    echo "LSTM_A norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_lstmA_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"

    echo "LSMT-B norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_lstmB_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"

    echo "GRU-A norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_gruA_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"

    echo "GRU-B norm"
    YAML_FILE="./experiments/$DATA_PATH/parameters_gruB_anomaly_norm.yaml"
    python run_exp_anomaly.py --config_file "$YAML_FILE"
done

cd bash