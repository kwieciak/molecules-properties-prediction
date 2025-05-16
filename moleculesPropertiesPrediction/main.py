import warnings
import torch
from data_loader import dataloader
from model import GNNwithMTL, trainer, tester, GNN
import time

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print(device)
    epochs = 51
    target_indices = [0, 1, 2, 3]
    batch_size = 32
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    dataset_usage_ratio = 0.001

    train_loader, val_loader, test_loader = dataloader.load_dataset(batch_size, train_ratio, val_ratio, test_ratio,
                                                                    target_indices, device, dataset_usage_ratio)

    modelGIN = GNNwithMTL.GIN(11, 64, target_indices).to(device)

    print('Dla batch = 32')
    start = time.time()
    # gin_train_loss32, gin_val_loss32, gin_train_target32, gin_train_y_target32 = trainer.train_epochs(epochs, modelGIN ,
    #                                                                                                   train_loader,
    #                                                                                                   val_loader,
    #                                                                                                   "GIN_model32.pt",device)
    gin_train_loss, gin_val_loss = trainer.train_epochs(epochs, modelGIN,
                                                        train_loader,
                                                        val_loader,
                                                        "GIN_model32.pt",
                                                        device)
    end = time.time()
    print('Time = ', end - start)

    return


main()
