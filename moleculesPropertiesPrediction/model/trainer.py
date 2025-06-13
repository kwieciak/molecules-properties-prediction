import torch

from utils.earlystopper import EarlyStopper


def train_gnn(loader, model, loss_fn, optimizer, device, task_weights=None):
    # switching into training mode
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        preds, targets = _prepare_preds_and_targets(batch, model, device)

        per_task_loss = loss_fn(preds, targets)

        # if task_weights is not None:
        # TODO: add implementation of task_weights handling

        loss = per_task_loss.mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_gnn(loader, model, loss_fn, device, task_weights=None):
    # switching into eval mode
    model.eval()
    total_loss = 0.0

    for batch in loader:
        preds, targets = _prepare_preds_and_targets(batch, model, device)

        per_task_loss = loss_fn(preds, targets)

        # if task_weights is not None:
        # TODO: add implementation of task_weights handling

        loss = per_task_loss.mean()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_epochs(epochs, model, train_loader, val_loader, path, device, optimizer, loss_fn, task_weights=None):
    early_stopper = EarlyStopper(patience=5, min_delta=0.05)

    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss = train_gnn(train_loader, model, loss_fn, optimizer, device)
        val_loss = eval_gnn(val_loader, model, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), path)

        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break

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

def _prepare_preds_and_targets(batch, model, device):
    batch = batch.to(device)
    batch.x = batch.x.float()
    batch.y = batch.y.float()

    preds = model(batch)

    batch_size = batch.y.shape[0]
    idx = torch.arange(batch_size, device=batch.y.device)
    targets = batch.y[idx, batch.r_target]

    return preds, targets
