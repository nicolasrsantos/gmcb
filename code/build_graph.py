import random
import networkx as nx
from collections import defaultdict
from utils import *
from args import *

VALID_DATASETS = ["20ng", "ohsumed", "R8", "R52", "mr", "ag_news", "SST1", "SST2", "TREC", "WebKB"]


def get_data(args):
    if args.dataset not in VALID_DATASETS:
        raise Exception("dataset not valid.\n supported datasets {VALID_DATASETS}")

    dataset_filename = os.path.join(args.cleaned_dir, args.dataset + ".clean.txt")
    info_filename = os.path.join(args.info_dir, args.dataset + ".txt")
    emb_filename = os.path.join(args.emb_dir, args.emb_file + ".txt")

    dataset = read_file(dataset_filename)
    train_test_info = read_file(info_filename)
    embs = read_embedding_file(emb_filename)

    assert len(dataset) == len(train_test_info)
    return dataset, train_test_info, embs


def get_word_nodes(dataset):
    word_nodes = set()
    for doc in dataset:
        doc_words = doc.split()
        word_nodes.update(doc_words)
    word_nodes = list(word_nodes)

    return word_nodes


def build_graph(dataset, y_map, doc_name, embs):
    graph_dict = {}
    word_nodes = get_word_nodes(dataset)
    word_id_map = {word:i for i, word in enumerate(word_nodes)}
    graph_dict["word_id_map"] = word_id_map

    G = nx.Graph()
    for doc_id, doc in enumerate(dataset):
        doc_words = doc.split()
        doc_id = "doc_" + str(doc_id)
        G.add_node(doc_id, bipartite=0)
        for word in doc_words:
            word_id = word_id_map[word]
            G.add_node(word_id, bipartite=1)
            G.add_edge(doc_id, word_id)
    print(f"dataset has {len(dataset)} docs and {len(get_word_nodes(dataset))} words")

    docs, words = [], []
    for edge in G.edges(data=True):
        if isinstance(edge[0], str):
            i, j = edge[0].split("_"), edge[1]
        else:
            i, j = edge[1].split("_"), edge[0]
        i = int(i[1])
        docs.append(i)
        words.append(j)
    graph_dict["edge_index"] = [docs, words]

    print("building feature vectors")
    x_doc = []
    for _ in range(len(dataset)):
        x_doc.append(np.random.uniform(-.01, .01, 300))
    graph_dict["x_doc"] = x_doc

    oov = create_rand_features(word_nodes)
    for key, value in oov.items():
        if key not in embs:
            embs[key] = value
    graph_dict["embs"] = embs

    x_word = []
    for i, word in enumerate(word_nodes):
        x_word.append(embs[word])
    graph_dict["x_word"] = x_word

    y = []
    for i in range(len(dataset)):
        doc_meta = doc_name[i].split('\t')
        label = doc_meta[2]
        y.append(y_map[label])
    graph_dict["y"] = y

    return graph_dict


def run(args):
    set_seed(42)

    dataset, train_test_info, embs = get_data(args)

    doc_name_list = []
    doc_train_list, doc_test_list = [], []
    for tti in train_test_info:
        doc_name_list.append(tti.strip())
        temp = tti.split()

        if temp[1].find("train") != -1:
            doc_train_list.append(tti.strip())
        if temp[1].find("test") != -1:
            doc_test_list.append(tti.strip())

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.shuffle(test_ids)

    ids = train_ids + test_ids

    # shuffle dataset
    shuffle_doc_name_list = []
    shuffle_dataset = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_dataset.append(dataset[int(id)])

    # Get labels
    y = []
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split("\t")
        y.append(temp[2])
    y_map = {label:i for i, label in enumerate(set(y))}

    train_size = len(train_ids)
    val_size = int(args.val_split * train_size)
    real_train_size = train_size - val_size

    masks_train = ids[0:real_train_size]
    masks_val = ids[real_train_size:real_train_size+val_size]
    masks_test = ids[train_size:]

    graph_dict = build_graph(
        shuffle_dataset, y_map, shuffle_doc_name_list, embs
    )
    graph_dict["masks"] = [masks_train, masks_val, masks_test]

    splits = np.empty(len(shuffle_dataset), dtype=object)
    splits[masks_train] = "train"
    splits[masks_val] = "val"
    splits[masks_test] = "test"

    if None in splits:
        raise ValueError("'None' found in splits list")
    graph_dict["splits"] = splits
    save_graph(graph_dict, args)


if __name__ == "__main__":
    args = args_build_graph()
    print(args)
    run(args)
