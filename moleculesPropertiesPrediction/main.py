import warnings

import torch
from data_loader import dataloader
from model import GNN, trainer, tester

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    train_loader, val_loader, test_loader = dataloader.load_dataset(64,0.7,0.1,0.2)
    model = GNN.GCN(64,11)
    return

main()