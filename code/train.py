import time
import datetime
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.nn import to_hetero, GIN, GAT, GraphSAGE
from torch_geometric.data.batch import Batch

from utils import *
from args import *
from neighbor_sampler import *


@torch.no_grad()
def init_params(train_loader, model, device):
    batch = next(iter(train_loader))
    batch = batch.to(device)
    model(batch.x_dict, batch.edge_index_dict)


def train(train_loader, model, optimizer, device):
    model.train()

    y_pred, y_true = [], []
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        batch = batch.to(device)
        batch_size = batch["doc"].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out["doc"][:batch_size]
        loss = F.cross_entropy(out, batch["doc"].y[:batch_size])
        preds = F.softmax(out, dim=-1)
        preds = preds.argmax(dim=-1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(batch["doc"].y[:batch_size].cpu().numpy())

        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        train_loss = total_loss / total_examples
    train_acc = accuracy_score(y_true, y_pred)
    train_f1 = f1_score(y_true, y_pred, average="macro")

    return train_loss, train_acc, train_f1


@torch.no_grad()
def eval(loader, model, device):
    model.eval()

    y_pred, y_true = [], []
    total_examples = total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch["doc"].batch_size

        out = model(batch.x_dict, batch.edge_index_dict)["doc"][:batch_size]
        loss = F.cross_entropy(out, batch["doc"].y[:batch_size])
        preds = F.softmax(out, dim=-1)
        preds = preds.argmax(dim=-1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(batch["doc"].y[:batch_size].cpu().numpy())

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        eval_loss = total_loss / total_examples
    eval_acc = accuracy_score(y_true, y_pred)
    eval_f1 = f1_score(y_true, y_pred, average="macro")

    return eval_loss, eval_acc, eval_f1


def experiment(train_loader, val_loader, test_loader, model, args):
    device = "cpu"
    if args.cuda and torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    model = model.to(device)
    init_params(train_loader, model, device) # warm-up for benchmarking

    if device == "cuda":
        torch.cuda.synchronize() # wait for warm-up to complete entirely

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    experiment_id = str(time.time())
    model_filename = os.path.join(args.models_dir, f"{args.dataset}_{experiment_id}.pt")
    bad_count, best_val_loss = 0, float('inf') # early stopping

    print("starting training")
    runtimes = []
    for epoch in range(1, args.n_epochs + 1):
        if bad_count == args.patience:
            print(f"early stopping on epoch {epoch}")
            break

        if device == "cuda":
            torch.cuda.synchronize()

        start_epoch = time.time()
        train_loss, train_acc, train_f1 = train(train_loader, model, optimizer, device)

        if device == "cuda":
            torch.cuda.synchronize()
        runtimes.append(time.time() - start_epoch)

        val_loss, val_acc, val_f1 = eval(val_loader, model, device)

        if epoch == 1 or epoch % args.epoch_log == 0:
            print(
                f"epoch {epoch}\tloss {train_loss:.4f}\tacc {train_acc:.4f}\tf1 {train_f1:.4f}"
                f"\tval_loss {val_loss:.4f}\tval_acc {val_acc:.4f}\tval_f1 {val_f1:.4f}"
            )

        if val_loss <= best_val_loss:
            bad_count = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_filename)
        else:
            bad_count += 1

    model.load_state_dict(torch.load(model_filename))
    test_loss, test_acc, test_f1 = eval(test_loader, model, device)
    print(f"test_loss {test_loss:.4f}\ttest_acc {test_acc:.4f}\ttest_f1 {test_f1:.4f}")

    mem_alloc = mem_res = 0
    if device == "cuda":
        mem_alloc = torch.cuda.max_memory_allocated()
        mem_res = torch.cuda.max_memory_reserved()

    return test_loss, test_acc, test_f1, mem_alloc, mem_res, np.sum(runtimes)


def check_batch(loader):
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
    n_id = torch.tensor(n_id, dtype=torch.long)

    input_id = []
    input_id.extend(last_batch["doc"].input_id)
    input_id.extend(penultimate_batch["doc"].input_id)
    input_id = np.array(input_id)
    input_id = torch.tensor(input_id, dtype=torch.long)

    batch_size = last_batch["doc"].batch_size + penultimate_batch["doc"].batch_size
    batch_size = torch.tensor(batch_size, dtype=torch.long)

    word_x = []
    word_x.extend(last_batch["word"].x)
    word_x.extend(penultimate_batch["word"].x)
    word_x = np.array(word_x)
    word_x = torch.tensor(word_x, dtype=torch.float)

    word_n_id = []
    word_n_id.extend(last_batch["word"].n_id)
    word_n_id.extend(penultimate_batch["word"].n_id)
    word_n_id = np.array(word_n_id)
    word_n_id = torch.tensor(word_n_id, dtype=torch.long)

    doc_2_w_last = last_batch["doc", "has_word", "word"].edge_index
    e_id_d2w_last = last_batch["doc", "has_word", "word"].e_id
    doc_2_w_penultimate = penultimate_batch["doc", "has_word", "word"].edge_index
    e_id_d2w_pen = penultimate_batch["doc", "has_word", "word"].e_id
    e_id_d2w = []
    e_id_d2w.extend(torch.tensor(np.array(e_id_d2w_last), dtype=torch.long))
    e_id_d2w.extend(torch.tensor(np.array(e_id_d2w_pen), dtype=torch.long))

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
    e_id_w2d.extend(torch.tensor(np.array(e_id_w2d_last), dtype=torch.long))
    e_id_w2d.extend(torch.tensor(np.array(e_id_w2d_pen), dtype=torch.long))

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


def run(args):
    set_seed(42)

    graph_dict = read_graph(args)
    graph = prepare_graph(graph_dict)

    train_nodes = ("doc", graph["doc"].train_mask)
    val_nodes = ("doc", graph["doc"].val_mask)
    test_nodes = ("doc", graph["doc"].test_mask)

    train_loader = NeighborLoader(
        graph,
        num_neighbors=[-1] * 2,
        input_nodes=train_nodes,
        batch_size=args.batch_size
    )
    # check for potential batches of size 1. since the mlps used on GIN
    # employ batch norm layers, we cannot batches with a single node.
    train_loader = check_batch(train_loader)

    val_loader = NeighborLoader(
        graph,
        num_neighbors=[-1] * 2,
        input_nodes=val_nodes,
        batch_size=args.batch_size
    )
    test_loader = NeighborLoader(
        graph,
        num_neighbors=[-1] * 2,
        input_nodes=test_nodes,
        batch_size=args.batch_size
    )

    match args.model.lower():
        case "sage":
            print("sage")
            model = get_model("sage", args.x_dim, args.hidden_dim, args.out_dim, args.dropout, 2)
        case "gin":
            print("gin")
            model = get_model("gin", args.x_dim, args.hidden_dim, args.out_dim, args.dropout, 2)
        case "gat":
            print("gat")
            model = get_model("gat", args.x_dim, args.hidden_dim, args.out_dim, args.dropout, 2, args.n_heads, True)

        case _:
            raise ValueError(f"unsupported gnn type. supported gnns are: GAT, GraphSAGE and GIN.")
    model = to_hetero(model, graph.metadata(), aggr="sum")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    accs, f1s, losses = [], [], []
    max_mem_alloc, max_mem_reserved, total_runtime = [], [], []
    for i in range(args.n_runs):
        print(f"run number: {i + 1}")
        loss, acc, f1, mem_alloc, mem_res, runtime = experiment(train_loader, val_loader, test_loader, model, args)

        losses.append(loss)
        accs.append(acc)
        f1s.append(f1)
        max_mem_alloc.append(mem_alloc)
        max_mem_reserved.append(mem_res)
        total_runtime.append(runtime)
        to_csv(loss, acc, f1, mem_res, mem_alloc, runtime, False, args)

    avg_loss = np.mean(losses)
    acc_avg, acc_std = np.mean(accs), np.std(accs)
    f1_avg, f1_std = np.mean(f1s), np.std(f1s)
    avg_mem_alloc = np.mean(max_mem_alloc)/1024**2
    avg_mem_res = np.mean(max_mem_reserved)/1024**2
    to_csv(avg_loss, acc_avg, f1_avg, avg_mem_res, avg_mem_alloc, np.sum(total_runtime), True, args, acc_std, f1_std)

    elapsed = str(datetime.timedelta(seconds=np.sum(total_runtime)))
    print(
        f"final acc: {acc_avg:.4f}\tacc_std: {acc_std:.4f}\n"
        f"final f1: {f1_avg:.4f}\tf1_std: {f1_std:.4f}\n"
        f"runtime: {elapsed}\n"
        f"avg memory allocated: {avg_mem_alloc:.2f} MB\n"
        f"avg memory reserved: {avg_mem_res:.2f} MB"
    )


if __name__ == "__main__":
    args = args_train()
    print(args)

    if args.dataset is None:
        raise ValueError("dataset arg was not set.")
    if args.out_dim is None:
        raise ValueError("output dimension was not set.")
    if args.coarsened and args.coarse_level is None:
        raise ValueError("input is a coarsened graph but coarse level was not set.")
    if args.coarsened and (args.partition is None or args.partition == "whole"):
        raise ValueError(
            "input is a coarsened graph, but the partition is incorrect. "
            f"partition is set to '{args.partition}', but 'docs', 'both' and 'words' implemented."
        )
    if not args.coarsened and args.partition != "whole":
        raise ValueError(
            "incorrect partition argument. only the 'whole' argument is "
            "supported for a non coarsened graph."
        )

    run(args)
