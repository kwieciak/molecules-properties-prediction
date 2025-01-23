import warnings
import torch
from data_loader import dataloader
from model import GNN, trainer, tester
import time

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    epochs = 50
    train_loader32, val_loader32, test_loader32 = dataloader.load_dataset(24,0.7,0.1,0.2)
    train_loader64, val_loader64, test_loader64 = dataloader.load_dataset(16,0.7,0.1,0.2)
    train_loader128, val_loader128, test_loader128 = dataloader.load_dataset(256,0.7,0.1,0.2)
    model = GNN.GCN(64, 11)

    print()
    print('Dla batch = 24')
    start1 = time.time()
    gcn_train_loss32, gcn_val_loss32, gcn_train_target32, gcn_train_y_target32 = trainer.train_epochs(epochs, model, train_loader32, test_loader32, "GCN_model32.pt")
    end1 = time.time()
    print('Time = ', end1 - start1)

    # start2 = time.time()
    # print()
    # print('Dla batch = 16')
    # gcn_train_loss64, gcn_val_loss64, gcn_train_target64, gcn_train_y_target64 = trainer.train_epochs(epochs, model,
    #                                                                                           train_loader64,
    #                                                                                           test_loader64,
    #                                                                                           "GCN_model64.pt")
    # end2 = time.time()
    # print('Time = ', end2 - start2)
    #
    #
    # start3 = time.time()
    # print()
    # print('Dla batch = 256')
    # gcn_train_loss128, gcn_val_loss128, gcn_train_target128, gcn_train_y_target128 = trainer.train_epochs(epochs, model,
    #                                                                                           train_loader128,
    #                                                                                           test_loader128,
    #                                                                                           "GCN_model128.pt")
    # end3 = time.time()
    # print('Time = ', end3 - start3)
    # return

main()