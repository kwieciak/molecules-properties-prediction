import math
import numpy as np
import torch

import utils.utils
from model import tester
from utils import utils
from utils.earlystopper import EarlyStopper


def train_gnn(loader, model, loss_fn, optimizer, device, task_weights=None):
    # switching into training mode
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        batch.x = batch.x.float()
        batch.y = batch.y.float()

        preds = model(batch)
        targets = batch.y.to(device)

        per_task_loss = loss_fn(preds, targets)

        if task_weights is not None:
            w = batch.y.new_tensor(task_weights)
            per_task_loss = per_task_loss * w

        loss = per_task_loss.mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(loader), model


@torch.no_grad()
def eval_gnn(loader, model, loss_fn, device, task_weights=None):
    # switching into eval mode
    model.eval()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        data.x = data.x.float()
        data.y = data.y.float()

        preds = model(data)
        targets = data.y

        per_task_loss = loss_fn(preds, targets)

        if task_weights is not None:
            w = data.y.new_tensor(task_weights)
            per_task_loss = per_task_loss * w

        loss = per_task_loss.mean()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_epochs(epochs, model, train_loader, val_loader, path, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss = torch.nn.MSELoss(reduction='none')
    early_stopper = EarlyStopper(patience=3, min_delta=0.05)

    train_target = np.empty(0)
    train_y_target = np.empty(0)
    train_loss = np.empty(epochs)
    val_loss = np.empty(epochs)
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss, model = train_gnn(train_loader, model, loss, optimizer, device)
        v_loss = eval_gnn(val_loader, model, loss, device)

        if early_stopper.early_stop(v_loss):
            print("Early stopping")
            break

        if v_loss < best_loss:
            torch.save(model.state_dict(), path)
            best_loss = v_loss
            
        for graph in train_loader:
            graph = graph.to(device)
            out = model(graph)
            if epoch == epochs - 1:
                # record truly vs predicted values for training data from last epoch
                train_target = np.concatenate((train_target, out.detach().cpu().numpy()[:, 0]))
                train_y_target = np.concatenate((train_y_target, graph.y.detach().cpu().numpy()))

        train_loss[epoch] = epoch_loss.detach().cpu().numpy()
        val_loss[epoch] = v_loss.detach().cpu().numpy()

        if epoch % 1 == 0:
            print("Epoch: " + str(epoch)
                  + ", Train loss: " + str(epoch_loss.item())
                  + ", Val loss: " + str(v_loss.item())
                  )
            # utils.print_gpu_memory()
            # utils.print_memory_usage()

        # these functions should not be called explicitly
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()

    return train_loss, val_loss, train_target, train_y_target
