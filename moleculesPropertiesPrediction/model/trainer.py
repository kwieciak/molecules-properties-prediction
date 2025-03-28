import math
import numpy as np
import torch

import utils.utils
from model import tester
from utils import utils
from utils.earlystopper import EarlyStopper


def train_gnn(loader, model, loss, optimizer, device):
    # swittching into training mode
    model.train()

    current_loss = 0
    for graph in loader:
        graph = graph.to(device)
        optimizer.zero_grad()
        graph.x = graph.x.float()
        graph.y = graph.y.float()

        out = model(graph)

        l = loss(out, torch.reshape(graph.y.to(device), (len(graph.y), 1)))
        current_loss += l / len(loader)

        l.backward()
        optimizer.step()

    return current_loss, model


@torch.no_grad()
def eval_gnn(loader, model, loss, device):
    # swittching into eval mode
    model.eval()

    val_loss = 0
    for graph in loader:
        graph = graph.to(device)
        out = model(graph)
        l = loss(out, torch.reshape(graph.y.to(device), (len(graph.y), 1)))
        val_loss += l / len(loader)
    return val_loss


def train_epochs(epochs, model, train_loader, val_loader, path, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss = torch.nn.MSELoss()
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
