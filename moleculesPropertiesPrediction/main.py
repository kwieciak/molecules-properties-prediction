import time
import warnings

import torch

import utils.utils
from data_loader import dataloader
from model import GNNwithMTL, trainer, tester

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#TODO: 1)generowanie wykresow 2)dokumentacja funkcji """ """

def main():
    print(device)
    epochs = 50
    batch_size = 24

    #qm9 targets:
    # 0 - dipole moment                                   10 - Free energy at 298.15K
    # 1 - isotropic polarizability                        11 - Heat capavity at 298.15K
    # 2 - Highest occupied molecular orbital energy       12 - Atomization energy at 0K
    # 3 - Lowest unoccupied molecular orbital energy      13 - Atomization energy at 298.15K
    # 4 - Gap between 2 and 3                             14 - Atomization enthalpy at 298.15K
    # 5 - Electronic spatial extent                       15 - Atomization free energy at 298.15K
    # 6 - Zero point vibrational energy                   16 - Rotational constant
    # 7 - Internal energy at 0K                           17 - Rotational constant
    # 8 - Internal energy at 298.15K                      18 - Rotational constant
    # 9 - Enthalpy at 298.15K

    #regression targets (tasks) selected to train the model
    train_r_targets1 = [1,2,3,4,5,6]
    train_r_targets2 = [1,2,4,5,6]

    #regression target (task) selected for model testing
    test_r_target1 = 0
    test_r_target2 = 0

    #merged lists to create the appropriate number of final_linear_layers in the model
    r_targets1 = train_r_targets1 + ([test_r_target1] if test_r_target1 not in train_r_targets1 else [])
    r_targets2 = train_r_targets2 + ([test_r_target2] if test_r_target2 not in train_r_targets2 else [])

    r_targets_weights1 = None
    r_targets_weights2 = None

    #how much of the dataset is taken for the task f.e. dataset_usage_ratio = 0.01 means that it is 1% of the entire qm9 dataset
    dataset_usage_ratio = 0.01

    #train, val, test subsets proportion f.e. train_ration=0.7 means that it is 70% of the loaded dataset
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2


    # train_loader1, val_loader1, test_loader1 = dataloader.load_dataset(batch_size, train_ratio, val_ratio, test_ratio,
    #                                                                 train_r_targets1, device, dataset_usage_ratio)
    train_loader2, val_loader2, test_loader2 = dataloader.load_dataset(batch_size, train_ratio, val_ratio, test_ratio,
                                                                       train_r_targets2, device, dataset_usage_ratio)

    #you can choose models: gin, gatv2cn, transformercn, gcn
    # modelGIN1 = GNNwithMTL.GIN(11, 64, r_targets1).to(device)
    modelGIN2 = GNNwithMTL.GIN(11, 64, r_targets2).to(device)

    # print('Dla batch = 24')
    # start = time.time()
    #
    # gin_train_loss1, gin_val_loss1 = trainer.train_epochs(epochs, modelGIN1,
    #                                                     train_loader1,
    #                                                     val_loader1,
    #                                                     "saved_models/GIN1.pt",
    #                                                     device, r_targets_weights1)
    # end = time.time()
    # print(f"Time = {end - start}")

    print('Dla batch = 24')
    start = time.time()

    gin_train_loss2, gin_val_loss2 = trainer.train_epochs(epochs, modelGIN2,
                                                        train_loader2,
                                                        val_loader2,
                                                        "GIN2.pt",
                                                        device, r_targets_weights2)
    end = time.time()
    print(f"Time = {end - start}")


    # metrics1, preds1, targets1 = tester.test_gnn(test_loader1, modelGIN1, test_r_target1, device)
    # print(f"Test RMSE: {metrics1['rmse']:.4f}, MAE: {metrics1['mae']:.4f}, R2: {metrics1['r2']:.4f}")

    metrics2, preds2, targets2 = tester.test_gnn(test_loader2, modelGIN2, test_r_target2, device)
    print(f"Test RMSE: {metrics2['rmse']:.4f}, MAE: {metrics2['mae']:.4f}, R2: {metrics2['r2']:.4f}")

    # utils.utils.plot_metric_comparison(metrics1, metrics2, "r2", "experiment with MTL", "experiment without MTL")
    # utils.utils.plot_metric_comparison(metrics1, metrics2, "rmse", "experiment with MTL", "experiment without MTL")
    # utils.utils.plot_metric_comparison(metrics1, metrics2, "mae", "experiment with MTL", "experiment without MTL")

    # utils.utils.plot_learning_curve(gin_train_loss1, gin_val_loss1, "experiment with MTL")
    utils.utils.plot_learning_curve(gin_train_loss2, gin_val_loss2, "transformercn, experiment without MTL")
    # utils.utils.plot_parity_plot(preds1, targets1, "experiment with MTL")
    utils.utils.plot_parity_plot(preds2, targets2, "transformercn, experiment without MTL")

    return


main()
