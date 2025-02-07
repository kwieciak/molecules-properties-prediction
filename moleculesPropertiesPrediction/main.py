import warnings
import torch
from data_loader import dataloader
from model import GNN, trainer, tester
import time

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(device)
    epochs = 51
    train_loader8, val_loader8, test_loader8 = dataloader.load_dataset(8,0.7,0.1,0.2,0.3)
    train_loader16, val_loader16, test_loader16 = dataloader.load_dataset(16,0.7,0.1,0.2,0.3)
    train_loader24, val_loader24, test_loader24 = dataloader.load_dataset(24,0.7,0.1,0.2,0.3)
    train_loader32, val_loader32, test_loader32 = dataloader.load_dataset(32,0.7,0.1,0.2,0.3)
    # modelGCN = GNN.GCN(11, 64).to(device)
    # modelGCN1 = GNN.GCN(11, 64).to(device)
    # modelGCN2 = GNN.GCN(11, 64).to(device)
    # modelGCN3 = GNN.GCN(11, 64).to(device)
    # modelGIN = GNN.GIN(11, 64).to(device)
    # modelGIN1 = GNN.GIN(11, 64).to(device)
    # modelGIN2 = GNN.GIN(11, 64).to(device)
    modelGIN3 = GNN.GIN(11, 64).to(device)
    # modelTransformCN = GNN.TransformerCN(11, 64).to(device)
    # modelTransformCN1 = GNN.TransformerCN(11, 64).to(device)
    # modelTransformCN2 = GNN.TransformerCN(11, 64).to(device)
    # modelTransformCN3 = GNN.TransformerCN(11, 64).to(device)
    # modelGATv2CN = GNN.Gatv2CN(11, 64).to(device)
    # modelGATv2CN1 = GNN.Gatv2CN(11, 64).to(device)
    # modelGATv2CN2 = GNN.Gatv2CN(11, 64).to(device)
    # modelGATv2CN3 = GNN.Gatv2CN(11, 64).to(device)

    #starttotaltime = time.time()
    # print("koniec tworzenia obiektow")
    # print("GCN:")
    # print()
    # print('Dla batch = 8')
    # start1 = time.time()
    # gcn_train_loss8, gcn_val_loss8, gcn_train_target8, gcn_train_y_target8 = trainer.train_epochs(epochs, modelGCN, train_loader8, val_loader8, "GCN_model8.pt", device)
    # end1 = time.time()
    # print('Time = ', end1 - start1)
    #
    # print()
    # print('Dla batch = 16')
    # start2 = time.time()
    # gcn_train_loss16, gcn_val_loss16, gcn_train_target16, gcn_train_y_target16 = trainer.train_epochs(epochs, modelGCN1,
    #                                                                                               train_loader16,
    #                                                                                               val_loader16,
    #                                                                                               "GCN_model16.pt", device)
    # end2 = time.time()
    # print('Time = ', end2 - start2)
    #
    # print()
    # print('Dla batch = 24')
    # start3 = time.time()
    # gcn_train_loss24, gcn_val_loss24, gcn_train_target24, gcn_train_y_target24 = trainer.train_epochs(epochs, modelGCN2,
    #                                                                                                   train_loader24,
    #                                                                                                   val_loader24,
    #                                                                                                   "GCN_model24.pt", device)
    # end3 = time.time()
    # print('Time = ', end3 - start3)
    #
    # print()
    # print('Dla batch = 32')
    # start4 = time.time()
    # gcn_train_loss32, gcn_val_loss32, gcn_train_target32, gcn_train_y_target32 = trainer.train_epochs(epochs, modelGCN3,
    #                                                                                                   train_loader32,
    #                                                                                                   val_loader32,
    #                                                                                                   "GCN_model32.pt", device)
    # end4 = time.time()
    # print('Time = ', end4 - start4)
    #
    # print("GIN:")
    # print()
    # print('Dla batch = 8')
    # start11 = time.time()
    # gin_train_loss8, gin_val_loss8, gin_train_target8, gin_train_y_target8 = trainer.train_epochs(epochs, modelGIN,
    #                                                                                               train_loader8,
    #                                                                                               val_loader8,
    #                                                                                               "GIN_model8.pt", device)
    # end11 = time.time()
    # print('Time = ', end11 - start11)
    #
    # print()
    # print('Dla batch = 16')
    # start22 = time.time()
    # gin_train_loss16, gin_val_loss16, gin_train_target16, gin_train_y_target16 = trainer.train_epochs(epochs, modelGIN1,
    #                                                                                                   train_loader16,
    #                                                                                                   val_loader16,
    #                                                                                                   "GIN_model16.pt", device)
    # end22 = time.time()
    # print('Time = ', end22 - start22)
    #
    # print()
    # print('Dla batch = 24')
    # start33 = time.time()
    # gin_train_loss24, gin_val_loss24, gin_train_target24, gin_train_y_target24 = trainer.train_epochs(epochs, modelGIN2,
    #                                                                                                   train_loader24,
    #                                                                                                   val_loader24,
    #                                                                                                   "GIN_model24.pt", device)
    # end33 = time.time()
    # print('Time = ', end33 - start33)

    print()
    print('Dla batch = 32')
    start44 = time.time()
    gin_train_loss32, gin_val_loss32, gin_train_target32, gin_train_y_target32 = trainer.train_epochs(epochs, modelGIN3,
                                                                                                      train_loader32,
                                                                                                      val_loader32,
                                                                                                      "GIN_model32.pt",device)
    end44 = time.time()
    print('Time = ', end44 - start44)

    # print("TransformCN:")
    # print()
    # print('Dla batch = 8')
    # start111 = time.time()
    # transform_train_loss8, transform_val_loss8, transform_train_target8, transform_train_y_target8 = trainer.train_epochs(epochs, modelTransformCN,
    #                                                                                               train_loader8,
    #                                                                                               val_loader8,
    #                                                                                               "transform_model8.pt", device)
    # end111 = time.time()
    # print('Time = ', end111 - start111)
    #
    # print()
    # print('Dla batch = 16')
    # start222 = time.time()
    # transform_train_loss16, transform_val_loss16, transform_train_target16, transform_train_y_target16 = trainer.train_epochs(epochs, modelTransformCN1,
    #                                                                                                   train_loader16,
    #                                                                                                   val_loader16,
    #                                                                                                   "transform_model16.pt", device)
    # end222 = time.time()
    # print('Time = ', end222 - start222)
    #
    # print()
    # print('Dla batch = 24')
    # start333 = time.time()
    # transform_train_loss24, transform_val_loss24, transform_train_target24, transform_train_y_target24 = trainer.train_epochs(epochs, modelTransformCN2,
    #                                                                                                   train_loader24,
    #                                                                                                   val_loader24,
    #                                                                                                   "transform_model24.pt", device)
    # end333 = time.time()
    # print('Time = ', end333 - start333)
    #
    # print()
    # print('Dla batch = 32')
    # start444 = time.time()
    # transform_train_loss32, transform_val_loss32, transform_train_target32, transform_train_y_target32 = trainer.train_epochs(epochs, modelTransformCN3,
    #                                                                                                   train_loader32,
    #                                                                                                   val_loader32,
    #                                                                                                   "transform_model32.pt", device)
    # end444 = time.time()
    # print('Time = ', end444 - start444)
    #
    # print("Gatv2CN:")
    # print()
    # print('Dla batch = 8')
    # start1111 = time.time()
    # gatv2_train_loss8, gatv2_val_loss8, gatv2_train_target8, gatv2_train_y_target8 = trainer.train_epochs(
    #     epochs, modelGATv2CN,
    #     train_loader8,
    #     val_loader8,
    #     "gatv2_model8.pt",
    # device)
    # end1111 = time.time()
    # print('Time = ', end1111 - start1111)
    #
    # print()
    # print('Dla batch = 16')
    # start2222 = time.time()
    # gatv2_train_loss16, gatv2_val_loss16, gatv2_train_target16, gatv2_train_y_target16 = trainer.train_epochs(
    #     epochs, modelGATv2CN1,
    #     train_loader16,
    #     val_loader16,
    #     "gatv2_model16.pt",
    # device)
    # end2222 = time.time()
    # print('Time = ', end2222 - start2222)
    #
    # print()
    # print('Dla batch = 24')
    # start3333 = time.time()
    # gatv2_train_loss24, gatv2_val_loss24, gatv2_train_target24, gatv2_train_y_target24 = trainer.train_epochs(
    #     epochs, modelGATv2CN2,
    #     train_loader24,
    #     val_loader24,
    #     "gatv2_model24.pt",
    # device)
    # end3333 = time.time()
    # print('Time = ', end3333 - start3333)
    #
    # print()
    # print('Dla batch = 32')
    # start4444 = time.time()
    # gatv2_train_loss32, gatv2_val_loss32, gatv2_train_target32, gatv2_train_y_target32 = trainer.train_epochs(
    #     epochs, modelGATv2CN3,
    #     train_loader32,
    #     val_loader32,
    #     "gatv2_model32.pt",
    # device)
    # end4444 = time.time()
    # print('Time = ', end4444 - start4444)

    #endtotaltime = time.time()
    #print('Time = ', endtotaltime - starttotaltime)

    return

main()