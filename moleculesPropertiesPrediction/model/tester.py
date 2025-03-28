import numpy as np
import torch


@torch.no_grad()
def test_gnn(loader, model, loss, device):
    loss = torch.nn.MSELoss()
    test_loss = 0
    test_target = np.empty(0)
    test_y_target = np.empty(0)
    for graph in loader:
        graph = graph.to(device)
        out = model(graph)
        l = loss(out, torch.reshape(graph.y.to(device), (len(graph.y), 1)))
        test_loss += l / len(loader)

        test_target = np.concatenate((test_target, out.detach().cpu().numpy()[:, 0]))
        test_y_target = np.concatenate((test_y_target, graph.y.detach().cpu().numpy()))

    return test_loss, test_target, test_y_target
