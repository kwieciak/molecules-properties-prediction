import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


@torch.no_grad()
def test_gnn(loader, model, test_task, device):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()
        batch.y = batch.y.float()

        batch_size = batch.y.shape[0]
        batch.r_target = torch.full((batch_size,), test_task, device=device)

        preds = model(batch)
        targets = batch.y[:, test_task]

        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    rmse = root_mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return {"rmse": rmse, "mae": mae, "r2": r2}
