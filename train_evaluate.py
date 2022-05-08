import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam

from utils.ppr_utils import select_from_csr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def train_and_evaluate(
    dataset,
    model,
    wandb,
    permute_masks=None,
    logger=None,
    runs=3,
    epochs=100,
    lr=0.005,
    weight_decay=0.0005,
    early_stopping=100,
    ppr=True,
    **kwargs,
):
    data = dataset[0]
    param_forward = kwargs.copy()
    if "ppr_matrix" in kwargs:
        ppr_matrix = param_forward.pop("ppr_matrix")
        ppr_vals = select_from_csr(ppr_matrix, data.edge_index).to(device)
        param_forward["ppr_vals"] = ppr_vals

    val_losses, train_accs, val_accs, test_accs, durations = [], [], [], [], []
    for r in range(runs):
        if r % 10 == 0:
            print(f"\nRun {r+1}/{runs}")

        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float("inf")
        train_acc, val_acc, test_acc = 0.0, 0.0, 0.0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data, ppr=ppr, **param_forward)
            eval_info = evaluate(model, data, ppr=ppr, **param_forward)
            eval_info["epoch"] = epoch

            if logger is not None:
                logger(eval_info)

            if eval_info["val_loss"] < best_val_loss:
                best_val_loss = eval_info["val_loss"]
                test_acc = eval_info["test_acc"]
                train_acc = eval_info["train_acc"]
                val_acc = eval_info["val_acc"]

            val_loss_history.append(eval_info["val_loss"])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(
                    val_loss_history[-(early_stopping + 1) : -1], dtype=torch.float64
                )
                if eval_info["val_loss"] > tmp.mean().item():
                    print(f"@@@@@@@ early_stopping: epoch {epoch}")
                    break
            if epoch % 10 == 0:
                wandb.log(eval_info)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        durations.append(t_end - t_start)

    val_loss, train_acc, val_acc, test_acc, duration = [
        tensor(i) for i in [val_losses, train_accs, val_accs, test_accs, durations]
    ]
    loss_mean, train_acc_mean, val_acc_mean, test_acc_mean, duration_mean = [
        i.mean().item() for i in [val_loss, train_acc, val_acc, test_acc, duration]
    ]
    test_acc_std = test_acc.std().item()
    wandb_log = {
        "val_loss": loss_mean,
        "train_acc": train_acc_mean,
        "val_acc": val_acc_mean,
        "test_acc": test_acc_mean,
        "test_acc_std": test_acc_std,
        "duration": duration_mean,
        "epoch_end": epoch,
    }
    print(
        "Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}".format(
            val_loss.mean().item(),
            test_acc.mean().item(),
            test_acc.std().item(),
            duration.mean().item(),
        )
    )
    return wandb_log


def train(
    model,
    optimizer,
    data,
    lambda1=1e-4,
    lambda2=1e-4,
    p=1,
    fused=0,
    ppr=True,
    **kwargs,
):
    model.train()
    optimizer.zero_grad()
    if ppr:
        out, (edge_index1, alpha1), (edge_index2, alpha2) = model(
            data.x, return_attention_weights=True, **kwargs
        )
        loss0 = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        reg_l1_1 = lambda1 * torch.norm(alpha1, int(p))
        reg_l1_2 = lambda2 * torch.norm(alpha2, int(p))
        loss = loss0 + reg_l1_1 + reg_l1_2
        if fused > 0:
            alpha1_norm = alpha1 / alpha1.sum()
            alpha2_norm = alpha2 / alpha2.sum()
            fused_l1 = fused * torch.norm(alpha1_norm - alpha2_norm, int(p))
            loss += fused_l1
    else:
        out = model(data.x, **kwargs)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()


def evaluate(model, data, ppr=True, **kwargs):
    model.eval()

    with torch.no_grad():
        if ppr:
            logits, (edge_index1, alpha1), (edge_index2, alpha2) = model(
                data.x, **kwargs
            )
        else:
            logits = model(data.x, **kwargs)

    outs = {}
    for key in ["train", "val", "test"]:
        mask = data["{}_mask".format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs["{}_loss".format(key)] = loss
        outs["{}_acc".format(key)] = acc

    return outs
