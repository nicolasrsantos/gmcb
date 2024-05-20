#!/bin/bash
datasets=("mr")
out=2
partitions=("both" "words" "docs")
bsz=64
hidden=256
gnns=("gat")
dropout=(0.2)
lr=0.001

for dataset in "${datasets[@]}"; do
	for gnn in "${gnns[@]}"; do
		for drop in "${dropout[@]}"; do
			python train.py --dataset $dataset --lr $lr --hidden_dim $hidden --out_dim $out --dropout $drop --model $gnn
			for partition in "${partitions[@]}"; do
				for i in $(seq 10 $END); do
					echo "running script - dataset $dataset - partition $partition - coarse_level $i"
					python train.py --dataset $dataset --lr $lr --hidden_dim $hidden --out_dim $out --coarse_level $i --coarsened --partition $partition --model $gnn --dropout $drop
				done
			done
		done
	done
done
