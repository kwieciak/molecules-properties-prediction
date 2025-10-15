import torch
import torch_geometric
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, accuracy_score, f1_score

from enums import enums
from model.trainer import compute_loss
from utils.utils import ensure_2d_tensor


@torch.no_grad()
def test_nn(loader, model, device, loss_fn, task_type, target_attr="y", test_task=None, num_classes=None):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        if isinstance(batch, torch_geometric.data.Batch):
            batch = batch.to(device)
            y_all = getattr(batch, target_attr)
            y_all = ensure_2d_tensor(y_all)

            preds = model(batch)

            if hasattr(batch, "r_target"):
                batch_size = y_all.shape[0]
                setattr(batch, "r_target", torch.full((batch_size,), test_task, device=device))

            if test_task is not None:
                targets = y_all[:, test_task]
            else:
                targets = y_all

        elif isinstance(batch, (list, tuple)):
            preds = ensure_2d_tensor(batch[0]).to(device)
            targets = ensure_2d_tensor(batch[1]).to(device)
            preds = model(preds)

        else:
            raise TypeError(f"Unsupported type {type(batch)}")

        per_batch_loss = compute_loss(loss_fn, preds, targets, task_type, num_classes)

        total_loss += per_batch_loss.sum().item()
        total_n += per_batch_loss.numel()

        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0).cpu()
    all_targets = torch.cat(all_targets, dim=0).cpu()

    if task_type == enums.TaskType.REGRESSION:
        if all_preds.dim() == 2 and all_preds.size(-1) == 1:
            all_preds = all_preds.squeeze(-1)
        rmse = root_mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    elif task_type in (enums.TaskType.MULTICLASS, enums.TaskType.BINARY):
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
