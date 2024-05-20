#!/bin/bash
datasets=("ohsumed" "R8" "R52" "mr" "SST1" "SST2" "TREC" "WebKB" "20ng")

for dataset in "${datasets[@]}"
do
    echo "running script - dataset $dataset"
    python build_graph.py --dataset $dataset
done
