#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

cd ../
echo "EGCNO norm"
python run_exp_anomaly.py --config_file ./experiments/parameters_egcn_o_anomaly_norm.yaml --debug

cd bash