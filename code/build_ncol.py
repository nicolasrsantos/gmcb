import pickle as pkl
from pathlib import Path
from collections import defaultdict
import os

from args import *

def to_interval(val, old_interval, new_interval):
    new_val = (((val-old_interval[0])*(new_interval[1]-new_interval[0]))//(old_interval[1]-old_interval[0]))+new_interval[0]
    return int(new_val)


def run(args):
    print("reading edge index")
    cur_graph_dir = os.path.join(args.graphs_dir, args.dataset, "whole", args.dataset)
    with open(cur_graph_dir + ".edge_index", "rb") as f:
        edge_index = pkl.load(f)

    print("converting data")
    doc, words = list(set(edge_index[0])), list(set(edge_index[1]))
    old_interval = [0, max(words)]
    new_interval = [len(doc), len(doc)+max(words)] # or len(doc)+len(words)-1
    print(old_interval, new_interval)
    word_int_map = {
        word:to_interval(word, old_interval, new_interval) for word in list(dict.fromkeys(words))
    }

    edge_list = []
    for i, j in zip(edge_index[0], edge_index[1]):
        edge_list.append([i, word_int_map[j], 1.0])

    ncol_dir = os.path.join(args.mfbn_dir, args.dataset)
    ncol_filename = os.path.join(ncol_dir, args.dataset + ".ncol")
    Path(ncol_dir).mkdir(parents=True, exist_ok=True)
    print(f"saving ncol to {ncol_filename}")
    with open(ncol_filename, "w") as f:
        for edge in edge_list:
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")

if __name__ == "__main__":
    args = args_coarse()
    print(args)

    if args.dataset is None:
        raise Exception("dataset was not specified")

    run(args)
