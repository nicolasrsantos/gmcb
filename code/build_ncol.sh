#!/bin/bash
datasets=("ohsumed" "R8" "R52" "mr" "SST1" "SST2" "TREC" "WebKB" "20ng")
mfbn_dir="../mfbn"

for dataset in "${datasets[@]}"
do
    echo "running script - dataset $dataset"
    python build_ncol.py --dataset $dataset \
	&& cp "data/graphs/$dataset/whole/$dataset.y" "$mfbn_dir/input/$dataset/" \
	&& cp "data/graphs/$dataset/whole/$dataset.splits" "$mfbn_dir/input/$dataset/" \
	&& cp "data/graphs/$dataset/whole/$dataset.x_doc" "$mfbn_dir/input/$dataset/" \
	&& cp "data/graphs/$dataset/whole/$dataset.x_word" "$mfbn_dir/input/$dataset/"
done
