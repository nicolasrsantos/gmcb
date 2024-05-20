import pickle as pkl
import json
from collections import defaultdict

from utils import *
from args import *

def to_interval(val, old_interval, new_interval):
    new_val = (((val-old_interval[0])*(new_interval[1]-new_interval[0]))/(old_interval[1]-old_interval[0]))+new_interval[0]
    return int(new_val)

set_seed(42)

def run(args):
    cur_dir = os.path.join(args.graphs_dir, args.dataset)
    with open(f"{cur_dir}/whole/{args.dataset}.embs", "rb") as f:
        word_embs = pkl.load(f)
    with open(f"{cur_dir}/whole/{args.dataset}.x_doc", "rb") as f:
        doc_embs = pkl.load(f)
    with open(f"{cur_dir}/whole/{args.dataset}.word_id_map", "rb") as f:
        word_id_map = pkl.load(f)
    with open(f"{cur_dir}/whole/{args.dataset}.y", "rb") as f:
        y = pkl.load(f)
    with open(f"{cur_dir}/whole/{args.dataset}.splits", "rb") as f:
        splits = pkl.load(f)

    coarsened_partition = ["both", "docs", "words"]
    for partition in coarsened_partition:
        print(f"parsing {partition}")
        graph_filename = os.path.join(cur_dir, partition, args.dataset)
        for coarse_level in range(1, args.max_level + 1):
            with open(f"{graph_filename}-{str(coarse_level)}.ncol", "r") as f:
                adj_list = f.readlines()
            with open(f"{graph_filename}-{str(coarse_level)}.membership", "r") as f:
                membership = f.readlines()
            with open(f"{graph_filename}-{str(coarse_level)}.weight", "r") as f:
                weight = f.readlines()
            with open(f"{graph_filename}-{str(coarse_level)}-info.json") as f:
                coarse_info = json.load(f)

            sv_map = defaultdict(list)
            membership = [int(el) for el in membership]
            for i, node in enumerate(membership):
                sv_map[node].append(i)

            old_int = [min(list(word_id_map.values())), max(list(word_id_map.values()))]
            new_int = [coarse_info["source_vertices"][0], len(membership)-1]

            id_word_map = {}
            for k, v in word_id_map.items():
                new_key = to_interval(v, old_int, new_int)
                id_word_map[new_key] = k

            x_doc_sv, x_word_sv, y_coarsened, masks_coarsened = [], [], [], [[],[],[]] # this is ugly, i know
            word_partition_start = coarse_info["coarsened_vertices"][0]
            for sv in sv_map.keys():
                sv_emb, y_, splits_ = [], [], []
                if sv < word_partition_start: # it's a doc
                    for v in sv_map[sv]:
                        sv_emb.append(doc_embs[v])
                        y_.append(y[v])
                        splits_.append(splits[v])

                    sv_emb = np.mean(sv_emb, axis=0)
                    x_doc_sv.append(np.array(sv_emb))

                    assert len(list(set(y_))) == 1
                    assert len(list(set(splits_))) == 1
                    y_coarsened.append(y_[0])

                    if splits_[0] == "train":
                        masks_coarsened[0].append(sv)
                    elif splits_[0] == "val":
                        masks_coarsened[1].append(sv)
                    elif splits_[0] == "test":
                        masks_coarsened[2].append(sv)
                else: # it's a word
                    for v in sv_map[sv]:
                        word = id_word_map[v]
                        sv_emb.append(word_embs[word])
                    sv_emb = np.mean(sv_emb, axis=0)
                    x_word_sv.append(np.array(sv_emb))
            x_doc_sv, x_word_sv = np.array(x_doc_sv), np.array(x_word_sv)

            src, tgt, edge_weight = [], [], []
            for edge in adj_list:
                edge = edge.split()
                i, j, w = int(edge[0]), int(edge[1]), int(edge[2])
                src.append(i)
                tgt.append(j)
                edge_weight.append(w)

            before = [min(tgt), max(tgt)]
            after = [0, coarse_info["coarsened_vertices"][1] - 1]
            print(before, after)

            tgt = [to_interval(v, before, after) for v in tgt]
            edge_index = [src, tgt]
            print(max(src), max(masks_coarsened[0]), max(masks_coarsened[1]), max(masks_coarsened[2]))
            print(min(src), min(masks_coarsened[0]), min(masks_coarsened[1]), min(masks_coarsened[2]))

            uniques = {}
            for edge in zip(edge_index[0], edge_index[1]):
                if edge in uniques:
                    uniques[edge] += 1
                else:
                    uniques[edge] = 1
            assert len(uniques) == coarse_info["coarsened_ecount"]

            with open(f"{graph_filename}-{str(coarse_level)}.edge_index", "wb") as f:
                pkl.dump(edge_index, f)
            with open(f"{graph_filename}-{str(coarse_level)}.x_word", "wb") as f:
                pkl.dump(x_word_sv, f)
            with open(f"{graph_filename}-{str(coarse_level)}.x_doc", "wb") as f:
                pkl.dump(x_doc_sv, f)
            with open(f"{graph_filename}-{str(coarse_level)}.y", "wb") as f:
                pkl.dump(y_coarsened, f)
            with open(f"{graph_filename}-{str(coarse_level)}.masks", "wb") as f:
                pkl.dump(masks_coarsened, f)
            with open(f"{graph_filename}-{str(coarse_level)}.edge_weight", "wb") as f:
                pkl.dump(edge_weight, f)

if __name__ == "__main__":
    args = args_coarse()
    print(args)

    if args.max_level == None:
        raise TypeError("please set max_level arg in order to process the coarsened graph")

    run(args)
