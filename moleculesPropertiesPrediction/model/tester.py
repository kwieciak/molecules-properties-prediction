import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


@torch.no_grad()
def test_gnn(loader, model, test_task, device, loss_fn):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        batch.y = batch.y.float()

        batch_size = batch.y.shape[0]
        batch.r_target = torch.full((batch_size,), test_task, device=device)

        preds = model(batch)
        targets = batch.y[:, test_task]

        per_task_loss = loss_fn(preds, targets)
        # loss = per_task_loss.mean()
        # total_loss += loss.item()

        total_loss += per_task_loss.sum().item()
        total_n += per_task_loss.numel()

        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0).cpu()
    all_targets = torch.cat(all_targets, dim=0).cpu()

    rmse = root_mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print("test_loss=", total_loss/total_n)

    return {"rmse": rmse, "mae": mae, "r2": r2}, all_preds, all_targets
