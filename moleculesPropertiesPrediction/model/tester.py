import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, accuracy_score, f1_score

from model.trainer import compute_loss


@torch.no_grad()
def test_gnn(loader, model, device, loss_fn, task_type, target_attr="y", test_task=None):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        y_all = getattr(batch, target_attr)
        if y_all.dim() == 1:
            y_all = y_all.unsqueeze(1)

        if hasattr(batch, "r_target"):
            batch_size = y_all.shape[0]
            setattr(batch, "r_target", torch.full((batch_size,), test_task, device=device))

        if test_task is not None:
            targets = y_all[:, test_task]
        else:
            targets = y_all

        preds = model(batch)

        per_batch_loss = compute_loss(loss_fn, preds, targets, task_type)

        total_loss += per_batch_loss.sum().item()
        total_n += per_batch_loss.numel()

        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0).cpu()
    all_targets = torch.cat(all_targets, dim=0).cpu()

    if task_type == "regression":
        if all_preds.dim() == 2 and all_preds.size(-1) == 1:
            all_preds = all_preds.squeeze(-1)
        rmse = root_mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    elif task_type in ("binary", "multiclass"):
        if all_preds.dim() != 2:
            raise ValueError(f"Multiclass expects preds of shape [N, C], got {tuple(all_preds.shape)}")
        pred_labels = all_preds.argmax(dim=-1)

        acc = accuracy_score(all_targets.numpy(), pred_labels.numpy())
        f1_macro = f1_score(all_targets.numpy(), pred_labels.numpy(), average="macro")
        f1_weighted = f1_score(all_targets.numpy(), pred_labels.numpy(), average="weighted")

        metrics = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    else:
        raise ValueError("Unknown task type")

    print("test_loss=", total_loss / total_n)

    return metrics, all_preds, all_targets
