import math
import numpy as np
import torch

from model import tester


def train_gnn(loader, model, loss, optimizer):
    #swittching into training mode
    model.train()

    current_loss = 0
    for graph in loader:
        optimizer.zero_grad()
        graph.x = graph.x.float()

        out = model(graph)

        l = loss(out, torch.reshape(graph.y, (len(graph.y), 1)))
        current_loss += l / len(loader)

        l.backward()
        optimizer.step()

    return current_loss, model

def train_epochs(epochs, model, train_loader, val_loader, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss = torch.nn.MSELoss()

    train_target = np.empty(0)
    train_y_target = np.empty(0)
    train_loss = np.empty(epochs)
    val_loss = np.empty(epochs)
    best_loss = math.inf

    for epoch in range(epochs):
        epoch_loss, model = train_gnn(train_loader, model, loss, optimizer)
        v_loss = tester.eval_gnn(val_loader, model, loss)
        if v_loss < best_loss:
            torch.save(model.state_dict(), path)
        for graph in train_loader:
            out = model(graph)
            if epoch == epochs - 1:
                # record truly vs predicted values for training data from last epoch
                train_target = np.concatenate((train_target, out.detach().numpy()[:, 0]))
                train_y_target = np.concatenate((train_y_target, graph.y.detach().numpy()))

        train_loss[epoch] = epoch_loss.detach().numpy()
        val_loss[epoch] = v_loss.detach().numpy()


        if epoch % 5 == 0:
            print("Epoch: " + str(epoch)
                + ", Train loss: " + str(epoch_loss.item())
                + ", Val loss: " + str(v_loss.item())
            )
    return train_loss, val_loss, train_target, train_y_target