import time
import warnings

import torch

import utils.utils
from data_loader import dataloader
from model import GNNwithMTL, trainer, tester

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO: dokumentacja funkcji """ """

def main():
    print(device)
    epochs = 200
    batch_size = 24
    start_index = 0

    # qm9 targets:
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

    # regression targets (tasks) selected to train the model
    train_r_targets1 = [11]
    train_r_targets2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]

    # regression target (task) selected for model testing
    test_r_target1 = 11
    test_r_target2 = 11

    # merged lists to create the appropriate number of final_linear_layers in the model
    r_targets1 = train_r_targets1 + ([test_r_target1] if test_r_target1 not in train_r_targets1 else [])
    r_targets2 = train_r_targets2 + ([test_r_target2] if test_r_target2 not in train_r_targets2 else [])

    r_targets_weights1 = None
    r_targets_weights2 = None

    # how much of the dataset is taken for the task f.e. dataset_usage_ratio = 0.01 means that it is 1% of the entire qm9 dataset
    dataset_usage_ratio = 0.1

    # train, val, test subsets proportion f.e. train_ration=0.7 means that it is 70% of the loaded dataset
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    train_loader1, val_loader1, test_loader1 = dataloader.load_dataset(batch_size, train_ratio, val_ratio, test_ratio,
                                                                       train_r_targets1, device, dataset_usage_ratio,
                                                                       start_index)
    train_loader2, val_loader2, test_loader2 = dataloader.load_dataset(batch_size, train_ratio, val_ratio, test_ratio,
                                                                       train_r_targets2, device, dataset_usage_ratio,
                                                                       start_index)

    # you can choose models: gin, gatv2cn, transformercn, gcn
    model1 = GNNwithMTL.GIN(11, 64, r_targets1).to(device)
    model2 = GNNwithMTL.Gatv2CN(11, 64, r_targets2).to(device)

    optimizer = torch.optim.Adam(model2.parameters(), lr=0.0005, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss(reduction='none')

    # print('Dla batch = 24')
    # start = time.time()
    #
    # gin_train_loss1, gin_val_loss1 = trainer.train_epochs(epochs, model1,
    #                                                     train_loader1,
    #                                                     val_loader1,
    #                                                     "saved_models/GNN1.pt",
    #                                                     device, optimizer, loss_fn, r_targets_weights1)
    # end = time.time()
    # print(f"Time = {end - start}")

    print('Dla batch = 24')
    start = time.time()

    gin_train_loss2, gin_val_loss2 = trainer.train_epochs(epochs, model2,
                                                          train_loader2,
                                                          val_loader2,
                                                          "saved_models/GNN2.pt",
                                                          device, optimizer, loss_fn, r_targets_weights2)
    end = time.time()
    print(f"Time = {end - start}")

    trainer.freeze_layers(model2, ['conv'])

    train_ratio_ft = 0.12
    val_ratio_ft = 0.05
    test_ratio_ft = 0.83
    train_r_targets_ft = [11]

    train_loader_ft, val_loader_ft, test_loader_ft = dataloader.load_dataset(batch_size, train_ratio_ft, val_ratio_ft,
                                                                       test_ratio_ft, train_r_targets_ft, device, 0.002, 14080)

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=0.0005,
                                    weight_decay=5e-4)

    gin_train_loss_ft, gin_val_loss_ft = trainer.train_epochs(epochs, model2,
                                                          train_loader_ft,
                                                          val_loader_ft,
                                                          "saved_models/GNN2FT.pt",
                                                          device, optimizer_ft, loss_fn, r_targets_weights2)

    # metrics1, preds1, targets1 = tester.test_gnn(test_loader1, model1, test_r_target1, device)
    # print(f"Test RMSE: {metrics1['rmse']:.4f}, MAE: {metrics1['mae']:.4f}, R2: {metrics1['r2']:.4f}")

    metrics2, preds2, targets2 = tester.test_gnn(test_loader_ft, model2, test_r_target2, device)
    print(f"Test r=11 RMSE: {metrics2['rmse']:.4f}, MAE: {metrics2['mae']:.4f}, R2: {metrics2['r2']:.4f}")

    metrics3, preds3, targets3 = tester.test_gnn(test_loader2, model2, 1, device)
    print(f"Test r=1 RMSE: {metrics3['rmse']:.4f}, MAE: {metrics3['mae']:.4f}, R2: {metrics3['r2']:.4f}")

    metrics4, preds4, targets4 = tester.test_gnn(test_loader2, model2, 2, device)
    print(f"Test r=2 RMSE: {metrics4['rmse']:.4f}, MAE: {metrics4['mae']:.4f}, R2: {metrics4['r2']:.4f}")

    metrics5, preds5, targets5 = tester.test_gnn(test_loader2, model2, 3, device)
    print(f"Test r=3 RMSE: {metrics5['rmse']:.4f}, MAE: {metrics5['mae']:.4f}, R2: {metrics5['r2']:.4f}")

    metrics6, preds6, targets6 = tester.test_gnn(test_loader2, model2, 4, device)
    print(f"Test r=4 RMSE: {metrics6['rmse']:.4f}, MAE: {metrics6['mae']:.4f}, R2: {metrics6['r2']:.4f}")

    metrics7, preds7, targets7 = tester.test_gnn(test_loader2, model2, 5, device)
    print(f"Test r=5 RMSE: {metrics7['rmse']:.4f}, MAE: {metrics7['mae']:.4f}, R2: {metrics7['r2']:.4f}")

    # utils.utils.plot_metric_comparison(metrics1, metrics2, "r2", "experiment with MTL", "experiment without MTL")
    # utils.utils.plot_metric_comparison(metrics1, metrics2, "rmse", "experiment with MTL", "experiment without MTL")
    # utils.utils.plot_metric_comparison(metrics1, metrics2, "mae", "experiment with MTL", "experiment without MTL")

    # utils.utils.plot_learning_curve(gin_train_loss1, gin_val_loss1, "experiment with MTL")
    # utils.utils.plot_parity_plot(preds1, targets1, "experiment with MTL")
    utils.utils.plot_learning_curve(gin_train_loss2, gin_val_loss2, "transformercn, experiment with MTL")
    utils.utils.plot_parity_plot(preds2, targets2, "transformercn r=0, experiment with MTL")

    utils.utils.plot_parity_plot(preds3, targets3, "transformercn r=1, experiment with MTL")

    utils.utils.plot_parity_plot(preds4, targets4, "transformercn r=2, experiment with MTL")

    utils.utils.plot_parity_plot(preds5, targets5, "transformercn r=3, experiment with MTL")

    utils.utils.plot_parity_plot(preds6, targets6, "transformercn r=4, experiment with MTL")

    utils.utils.plot_parity_plot(preds7, targets7, "transformercn r=5, experiment with MTL")

    return


main()
