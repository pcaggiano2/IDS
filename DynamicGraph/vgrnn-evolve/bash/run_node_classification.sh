#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
cd ../
python run_exp.py --config_file ./experiments/parameters_elliptic_egcn_o.yaml --debug

cd bash