import warnings
import torch
from data_loader import dataloader
from model import GNN, trainer, tester
import time

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    epochs = 50
    train_loader24, val_loader24, test_loader24 = dataloader.load_dataset(24,0.7,0.1,0.2)
    train_loader32, val_loader32, test_loader32 = dataloader.load_dataset(16,0.7,0.1,0.2)
    modelGCN = GNN.GCN(64, 11)
    modelTransformCN = GNN.TransformerCN(11, 64)
    modelGATv2CN = GNN.Gatv2CN(11, 64)
    modelGIN = GNN.GIN(11, 64)

    print()
    print('Dla batch = 24')
    start1 = time.time()
    gcn_train_loss24, gcn_val_loss24, gcn_train_target24, gcn_train_y_target24 = trainer.train_epochs(epochs, modelGIN, train_loader24, test_loader24, "GCN_model24.pt")
    end1 = time.time()
    print('Time = ', end1 - start1)

    start2 = time.time()
    print()
    print('Dla batch = 32')
    gcn_train_loss32, gcn_val_loss32, gcn_train_target32, gcn_train_y_target32 = trainer.train_epochs(epochs, modelGIN,train_loader32, test_loader32, "GCN_model32.pt")
    end2 = time.time()
    print('Time = ', end2 - start2)

    return

main()