import torch
import torch.nn.functional as fun
import torch_geometric

from config import timestamp
from enums import enums
from utils.earlystopper import EarlyStopper
from utils.utils import ensure_folder, ensure_2d_tensor


def train_nn(loader, model, loss_fn, optimizer, device, task_type, target_attr="y", num_classes=None,
             task_weights=None):
    # switching into training mode
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        optimizer.zero_grad()
        preds, targets = _prepare_preds_and_targets(batch, model, device, target_attr)

        per_batch_loss = compute_loss(loss_fn, preds, targets, task_type, num_classes)

        # if task_weights is not None:
        # TODO: add implementation of task_weights handling

        loss = per_batch_loss.mean()

        loss.backward()
        optimizer.step()

        total_loss += per_batch_loss.sum().item()
        total_n += per_batch_loss.numel()

    return total_loss / total_n


@torch.no_grad()
def eval_nn(loader, model, loss_fn, device, task_type, target_attr="y", num_classes=None, task_weights=None):
    # switching into eval mode
    model.eval()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        preds, targets = _prepare_preds_and_targets(batch, model, device, target_attr)

        per_batch_loss = compute_loss(loss_fn, preds, targets, task_type, num_classes)

        # if task_weights is not None:
        # TODO: add implementation of task_weights handling

        total_loss += per_batch_loss.sum().item()
        total_n += per_batch_loss.numel()

    return total_loss / total_n


def train_epochs(epochs, model, train_loader, val_loader, filename, device, optimizer, loss_fn, scheduler, task_type,
                 target_attr, num_classes=None,
                 task_weights=None):
    ensure_folder(f"results/{timestamp}/saved_models")
    early_stopper = EarlyStopper(patience=15, min_delta=0.0005, path=f"results/{timestamp}/saved_models/{filename}")

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_nn(train_loader, model, loss_fn, optimizer, device, task_type, target_attr, num_classes)
        val_loss = eval_nn(val_loader, model, loss_fn, device, task_type, target_attr, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}",
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        try:
            scheduler.step(val_loss)
        except TypeError:
            scheduler.step()

        if early_stopper.check(val_loss, model):
            print("Early stopping")
            break
    early_stopper.load_best(model, device=device)

    return train_losses, val_losses


def freeze_layers(model, prefix_list):
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in prefix_list):
            for param in module.parameters():
                param.requires_grad = False


def unfreeze_layers(model, prefix_list):
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in prefix_list):
            for param in module.parameters():
                param.requires_grad = True


def freeze_features(model, exceptions_list):
    for param in model.parameters():
        param.requires_grad = False

    for exception in exceptions_list:
        exception_str = str(exception)
        if exception_str in model.task_heads:
            for param in model.task_heads[exception_str].parameters():
                param.requires_grad = True


def unfreeze_features(model):
    for param in model.parameters():
        param.requires_grad = True


def _prepare_preds_and_targets(batch, model, device, target_attr):
    if isinstance(batch, torch_geometric.data.Batch):
        batch = batch.to(device)
        y_all = getattr(batch, target_attr)
        y_all = ensure_2d_tensor(y_all)

        preds = model(batch)

        if hasattr(batch, "r_target"):
            batch_size = y_all.shape[0]
            idx = torch.arange(batch_size, device=device)
            targets = y_all[idx, getattr(batch, "r_target")]
        else:
            targets = y_all
        return preds, targets

    elif isinstance(batch, (list, tuple)):
        preds = ensure_2d_tensor(batch[0]).to(device)
        targets = ensure_2d_tensor(batch[1]).to(device)
        preds = model(preds)
        return preds, targets

    else:
        raise TypeError(f"Unsupported type {type(batch)}")


def compute_loss(loss_fn, preds, targets, task_type, num_classes):
    if task_type == enums.TaskType.REGRESSION:
        if preds.dim() == 2 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)

        if targets.dim() > 1 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        return loss_fn(preds, targets)

    elif task_type in (enums.TaskType.MULTICLASS, enums.TaskType.BINARY):
        if preds.dim() != 2:
            raise ValueError(f"Multiclass expects preds of shape [N, C], got {tuple(preds.shape)}")
        targets = fun.one_hot(targets.squeeze(1).long(), num_classes=num_classes).float()
        return loss_fn(preds, targets)
    else:
        raise ValueError("Unknown task type")
