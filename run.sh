#!/bin/bash

# Load python virtualenv
virtualenv --python=/usr/bin/python3.8 venv
source venv/bin/activate
pip3.8 install -r requirements.txt

# Set GPU device ID
export CUDA_VISIBLE_DEVICES=-1

# Begin experiment
for SEED in {1..20}
do
    python3.8 main.py \
    --seed $SEED \
    --config "ibs.yaml" \
    --prefix ""
done
