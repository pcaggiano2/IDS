#!/bin/bash
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

echo "Cuda: $CUDA"
echo "MODEL: $MODEL"
echo "Normalize: $NORMA"
export CUDA_VISIBLE_DEVICES=${CUDA}

cd ../

if [ "$MODEL" = "EGCNO" ]; then
    echo "EGCN-O."
    #---- EGCN-O ----#
    if [ "$NORMA" = "True" ]; then
        echo "Normalized"
        #---- EGCN-O ----#
        python run_exp_anomaly.py --config_file ./experiments/parameters_egcn_o_anomaly_norm.yaml
    else
        echo "No-Normalized"
        python run_exp_anomaly.py --config_file ./experiments/parameters_egcn_o_anomaly_no_norm.yaml
    fi
fi

if [ "$MODEL" = "EGCNH" ]; then
    echo "EGCN-H."
    #---- EGCN-O ----#
    if [ "$NORMA" = "True" ]; then
        echo "Normalized"
        #---- EGCN-O ----#
        python run_exp_anomaly.py --config_file ./experiments/parameters_egcn_h_anomaly_norm.yaml
    else
        echo "No-Normalized"
        python run_exp_anomaly.py --config_file ./experiments/parameters_egcn_h_anomaly_no_norm.yaml
    fi
fi

if [ "$MODEL" = "LSTM" ]; then
    echo "LSTM"
    if [ "$NORMA" = "True" ]; then
        echo "Normalized"
        #---- GRU ----#
        python run_exp_anomaly.py --config_file ./experiments/parameters_lstm_anomaly_norm.yaml
    else
        echo "No-Normalized"
        python run_exp_anomaly.py --config_file ./experiments/parameters_lstm_anomaly_no_norm.yaml 
    fi
fi

if [ "$MODEL" = "GRU" ]; then
    echo "GRU"
    if [ "$NORMA" = "True" ]; then
        echo "Normalized"
        #---- GRU ----#
        python run_exp_anomaly.py --config_file ./experiments/parameters_gru_anomaly_norm.yaml
    else
        echo "No-Normalized"
        python run_exp_anomaly.py --config_file ./experiments/parameters_gru_anomaly_no_norm.yaml 
    fi
fi

cd bash