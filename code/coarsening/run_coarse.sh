#!/bin/bash
datasets=("R52" "ohsumed" "mr" "TREC" "SST1" "SST2" "WebKB" "20ng")
partitions=("docs" "words" "both")

for dataset in "${datasets[@]}"; do
	for partition in "${partitions[@]}"; do
		echo "running script - dataset $dataset - partition $partition"
		python coarse.py -cnf "input/$dataset/$dataset-emb_sim-gbm-$partition.json"
	done
done
