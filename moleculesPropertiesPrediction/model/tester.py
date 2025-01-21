import numpy as np
import torch

def eval_gnn(loader, model, loss):
    #swittching into eval mode
    model.eval()

    val_loss = 0
    for graph in loader:
        out = model(graph)
        l = loss(out, torch.reshape(graph.y, (len(graph.y), 1)))
        val_loss += l / len(loader)
    return val_loss

@torch.no_grad()
def test_gnn(loader, model, loss):
    loss = torch.nn.MSELoss()
    test_loss = 0
    test_target = np.empty(0)
    test_y_target = np.empty(0)
    for graph in loader:
        out = model(graph)
        l = loss(out, torch.reshape(graph.y, (len(graph.y), 1)))
        test_loss += l / len(loader)

        test_target = np.concatenate((test_target, out.detach().numpy()[:, 0]))
        test_y_target = np.concatenate((test_y_target, graph.y.detach().numpy()))

    return test_loss, test_target, test_y_target