import os
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
from torch.nn import ReLU, Dropout
from torch_geometric.data import HeteroData
from torch_geometric.nn import Sequential, Linear, SAGEConv, GINConv, GATv2Conv, MLP, GATConv
from torch.nn import functional as F

def read_file(filename):
    file_content = []
    with open(filename, "r") as f:
        for line in f.readlines():
            file_content.append(line)
    print(f"read a file with {len(file_content)} elements")

    return file_content


def read_embedding_file(filename):
    embeddings = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            data = line.split()
            word = data[0]
            emb_vec = [float(i) for i in data[1:]]
            embeddings[word] = emb_vec
    print(f"read an embedding file with {len(list(embeddings.keys()))} elements")

    return embeddings


def create_rand_features(data, low=-0.01, high=0.01, dim=300):
    random_features = {}

    if isinstance(data, dict):
        data = list(data.keys())
    elif isinstance(data, set):
        data = list(data)

    for i in range(len(data)):
        random_features[data[i]] = np.random.uniform(low, high, dim)

    return random_features


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_graph(graph_dict, args):
    cur_graph_dir = os.path.join(args.graphs_dir, args.dataset, "whole")
    save_str = os.path.join(cur_graph_dir, args.dataset)
    Path(cur_graph_dir).mkdir(parents=True, exist_ok=True)

    print(f"saving {args.dataset}'s graph information to : {cur_graph_dir}")
    with open(save_str + ".edge_index", "wb") as f:
        pkl.dump(graph_dict["edge_index"], f)
    with open(save_str + ".x_doc", "wb") as f:
        pkl.dump(graph_dict["x_doc"], f)
    with open(save_str + ".x_word", "wb") as f:
        pkl.dump(graph_dict["x_word"], f)
    with open(save_str + ".y", "wb") as f:
        pkl.dump(graph_dict["y"], f)
    with open(save_str + ".masks", "wb") as f:
        pkl.dump(graph_dict["masks"], f)
    with open(save_str + ".embs", "wb") as f:
        pkl.dump(graph_dict["embs"], f)
    with open(save_str + ".word_id_map", "wb") as f:
        pkl.dump(graph_dict["word_id_map"], f)
    with open(save_str + ".splits", "wb") as f:
        pkl.dump(graph_dict["splits"], f)
    print(f"{args.dataset}'s graph data saved")


def nx_to_edge_idx(G):
    source = []
    target = []

    for edge in G.edges(data=True):
        if isinstance(edge[0], str):
            i, j = edge[0].split("_"), edge[1]
        else:
            i, j = edge[1].split("_"), edge[0]
        i = int(i[1])
        source.append(i)
        target.append(j)
        source.append(j)
        target.append(i)
    edge_index = [source, target]

    return edge_index


def read_graph(args):
    cur_graph = os.path.join(args.graphs_dir, args.dataset, args.partition, args.dataset)

    if args.coarsened:
        cur_graph = cur_graph + f"-{str(args.coarse_level)}"
    elif not args.coarsened and args.partition != "whole":
        raise ValueError(
            "incorrect partition argument. only the 'whole' argument is\n"
            "supported for a non coarsened graph."
        )

    print(f"reading cur_graph {cur_graph}")
    graph_dict = {}
    with open(f"{cur_graph}.edge_index", "rb") as f:
        graph_dict["edge_index"] = pkl.load(f)
    with open(f"{cur_graph}.x_word", "rb") as f:
        graph_dict["x_word"] = pkl.load(f)
    with open(f"{cur_graph}.x_doc", "rb") as f:
        graph_dict["x_doc"] = pkl.load(f)
    with open(f"{cur_graph}.y", "rb") as f:
        graph_dict["y"] = pkl.load(f)
    with open(f"{cur_graph}.masks", "rb") as f:
        graph_dict["masks"] = pkl.load(f)

    return graph_dict

def prepare_graph(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)

    x_doc = np.array(graph_dict["x_doc"])
    x_doc = torch.tensor(x_doc, dtype=torch.float)
    x_word = np.array(graph_dict["x_word"])
    x_word = torch.tensor(x_word, dtype=torch.float)
    y = torch.tensor(graph_dict["y"], dtype=torch.long)

    n_doc_nodes = len(y)
    train_idx, val_idx, test_idx = graph_dict["masks"][0], graph_dict["masks"][1], graph_dict["masks"][2]
    train_mask = torch.zeros(n_doc_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_doc_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_doc_nodes, dtype=torch.bool)
    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1

    data = HeteroData()
    data["doc"].x = x_doc
    data["doc"].y = y
    data["doc"].train_mask = train_mask
    data["doc"].val_mask = val_mask
    data["doc"].test_mask = test_mask
    data["word"].x = x_word
    data["doc", "has_word", "word"].edge_index = edge_index
    data["word", "is_in_doc", "doc"].edge_index = edge_index.flip(0)

    return data

def check_graph_properties(G):
    print("checking graph properties")
    print(f"\tGraph has isolated nodes: {G.has_isolated_nodes()}")
    print(f"\tGraph has self loops: {G.has_self_loops()}")
    print(f"\tGraph is undirected: {G.is_undirected()}")


def to_csv(loss, acc, f1, mem_res, mem_alloc, runtime, is_avg, args, acc_std=None, f1_std=None):
    Path("results/").mkdir(parents=True, exist_ok=True)

    csv = f"{args.model},{args.lr},{args.batch_size},{args.dropout},{args.coarse_level},"
    csv = csv + f"{args.partition},{args.x_dim},{args.hidden_dim},{mem_res:4f},"
    csv = csv + f"{mem_alloc:4f},{runtime:4f},{loss:.4f},{acc:.4f},{f1:.4f}"

    if is_avg:
        csv = csv + f",{acc_std:4f},{f1_std:4f}\n"
        csv_file = f"results/{args.dataset}_avg_results.csv"
    else:
        csv = csv + "\n"
        csv_file = f"results/{args.dataset}_results.csv"

    with open(csv_file, "a") as f:
        f.write(csv)

def get_model(model, input_dim, hidden_dim, out_dim, dropout, num_layers=2, num_heads=4, v2=False):
    if model.lower() == "gat" and num_heads is None:
        raise ValueError("number of heads for GAT not defined.")

    match model.lower():
        case "sage":
            convs = [SAGEConv((-1, -1), hidden_dim) for _ in range(num_layers)]
        case "gin":
            mlps = [
                MLP(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=hidden_dim, num_layers=2),
                MLP(in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=out_dim, num_layers=2)
            ]
            convs = [GINConv(mlps[i]) for i in range(num_layers)]
        case "gat":
            gat = GATConv
            if v2:
                gat = GATv2Conv

            convs = [gat((-1, -1), hidden_dim, add_self_loops=False, heads=num_heads) for _ in range(num_layers)]
        case _:
            raise ValueError("unsupported gnn architecture. only 'gat', 'sage', and 'gin' are supported")

    model = Sequential("x, edge_index", [
        (convs[0], "x, edge_index -> x"),
        ReLU(inplace=True),
        (convs[1], "x, edge_index -> x"),
        ReLU(inplace=True),
        (Dropout(dropout), "x -> x"),
        (Linear(-1, out_dim), "x -> x"),
    ])
    return model

def check_batch(loader):
    # Warning: this code is very ugly. Proceed at your own risk :)
    batches = []
    num_batches = 0
    for batch in loader:
        batches.append(batch)
        num_batches += 1
    penultimate_batch = batches[-2]
    last_batch = batches[-1]

    # no last batch with batch_size 1 yay
    if last_batch["doc"].batch_size > 1:
        return loader

    x = []
    x.extend(last_batch["doc"].x)
    x.extend(penultimate_batch["doc"].x)
    x = np.array(x)
    x = torch.tensor(x, dtype=torch.float)

    y = []
    y.extend(last_batch["doc"].y)
    y.extend(penultimate_batch["doc"].y)
    y = np.array(y)
    y = torch.tensor(y, dtype=torch.long)

    train_mask = []
    train_mask.extend(last_batch["doc"].train_mask)
    train_mask.extend(penultimate_batch["doc"].train_mask)
    train_mask = np.array(train_mask)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)

    val_mask = []
    val_mask.extend(last_batch["doc"].val_mask)
    val_mask.extend(penultimate_batch["doc"].val_mask)
    val_mask = np.array(val_mask)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)

    test_mask = []
    test_mask.extend(last_batch["doc"].test_mask)
    test_mask.extend(penultimate_batch["doc"].test_mask)
    test_mask = np.array(test_mask)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    n_id = []
    n_id.extend(last_batch["doc"].n_id)
    n_id.extend(penultimate_batch["doc"].n_id)
    n_id = np.array(n_id)

    input_id = []
    input_id.extend(last_batch["doc"].input_id)
    input_id.extend(penultimate_batch["doc"].input_id)
    input_id = np.array(input_id)

    batch_size = last_batch["doc"].batch_size + penultimate_batch["doc"].batch_size

    word_x = []
    word_x.extend(last_batch["word"].x)
    word_x.extend(penultimate_batch["word"].x)
    word_x = np.array(word_x)

    word_n_id = []
    word_n_id.extend(last_batch["word"].n_id)
    word_n_id.extend(penultimate_batch["word"].n_id)
    word_n_id = np.array(word_n_id)

    doc_2_w_last = last_batch["doc", "has_word", "word"].edge_index
    e_id_d2w_last = last_batch["doc", "has_word", "word"].e_id
    doc_2_w_penultimate = penultimate_batch["doc", "has_word", "word"].edge_index
    e_id_d2w_pen = penultimate_batch["doc", "has_word", "word"].e_id
    e_id_d2w = []
    e_id_d2w.extend(np.array(e_id_d2w_last))
    e_id_d2w.extend(np.array(e_id_d2w_pen))

    doc_2_w = [[el for el in doc_2_w_last[0]], [el for el in doc_2_w_last[1]]]
    tmp = [el + 1 for el in doc_2_w_penultimate[0]]
    doc_2_w[0].extend(tmp)
    tmp = [el + 1 for el in doc_2_w_penultimate[1]]
    doc_2_w[1].extend(tmp)
    doc_2_w = torch.tensor(doc_2_w, dtype=torch.long)

    w_2_doc_last = last_batch["word", "is_in_doc", "doc"].edge_index
    e_id_w2d_last = last_batch["word", "is_in_doc", "doc"].e_id
    w_2_doc_penultimate = penultimate_batch["word", "is_in_doc", "doc"].edge_index
    e_id_w2d_pen = penultimate_batch["word", "is_in_doc", "doc"].e_id
    e_id_w2d = []
    e_id_w2d.extend(np.array(e_id_w2d_last))
    e_id_w2d.extend(np.array(e_id_w2d_pen))

    w_2_doc = [[w_2_doc_last[0]], [w_2_doc_last[1]]]
    tmp = [el + 1 for el in w_2_doc_penultimate[0]]
    w_2_doc[0].extend(tmp)
    tmp = [el + 1 for el in w_2_doc_penultimate[1]]
    w_2_doc[1].extend(tmp)
    w_2_doc = torch.tensor(w_2_doc, dtype=torch.long)


    hdata = HeteroData()
    hdata["doc"].x = x
    hdata["doc"].y = y
    hdata["doc"].train_mask = train_mask
    hdata["doc"].val_mask = val_mask
    hdata["doc"].test_mask = test_mask
    hdata["doc"].n_id = n_id
    hdata["doc"].input_id = input_id
    hdata["doc"].batch_size = batch_size
    hdata["word"].x = word_x
    hdata["word"].n_id = word_n_id
    hdata["word", "is_in_doc", "doc"].edge_index = w_2_doc
    hdata["word", "is_in_doc", "doc"].e_id = e_id_w2d
    hdata["doc", "has_word", "word"].edge_index = doc_2_w
    hdata["doc", "has_word", "word"].e_id = e_id_d2w

    final_batches = []
    for i in range(num_batches - 2):
        final_batches.append(batches[i])
    final_batches.append(hdata)
    return final_batches
