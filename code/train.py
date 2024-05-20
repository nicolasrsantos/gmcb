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
