import warnings
import torch

import utils.utils
from data_loader import dataloader
from model import GNNwithMTL, trainer, tester, GNN
import time


warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print(device)
    epochs = 100
    target_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #jakie taski sa wybierane do trenowania #trenowac na od 0 do 17 a testoweac na 18 vs trenowany tylko na 18
    test_task = 0 #jaki task jest wybieramy do testu
    task_indices = target_indices + ([test_task] if test_task not in target_indices else []) #sklejone listy aby w modelu utworzyc odpowiednia ilosc final_linear_layer
    task_weights = None
    batch_size = 24
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    dataset_usage_ratio = 0.05

    train_loader, val_loader, test_loader = dataloader.load_dataset(batch_size, train_ratio, val_ratio, test_ratio,
                                                                    target_indices, device, dataset_usage_ratio)

    modelGIN = GNNwithMTL.GIN(11, 64, task_indices).to(device)

    print('Dla batch = 32')
    start = time.time()
    # gin_train_loss32, gin_val_loss32, gin_train_target32, gin_train_y_target32 = trainer.train_epochs(epochs, modelGIN ,
    #                                                                                                   train_loader,
    #                                                                                                   val_loader,
    #                                                                                                   "saved_models/GIN.pt",device)
    gin_train_loss, gin_val_loss = trainer.train_epochs(epochs, modelGIN,
                                                        train_loader,
                                                        val_loader,
                                                        "saved_models/GIN.pt",
                                                        device, task_weights)
    end = time.time()
    print('Time = ', end - start)
    metrics = tester.test_gnn(test_loader, modelGIN, test_task, device)
    print(f"Test RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")

    return


main()
