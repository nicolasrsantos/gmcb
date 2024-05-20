#!/bin/bash
datasets=("ohsumed" "R8" "R52" "mr" "SST1" "SST2" "TREC" "WebKB" "20ng")
partitions=("both" "docs" "words")
max_level=10

for dataset in "${datasets[@]}"; do
	for partition in "${partitions[@]}"; do
		echo "running script - dataset $dataset"
		python parse_coarse.py --dataset $dataset --max_level $max_level
	done
done
