import time
import warnings

import torch

from data_loader.dataloader import load_dataset
from model import GNNwithMTL, trainer, tester
from utils.utils import get_timestamp, save_loss_to_csv, plot_parity_plot, plot_learning_curve, \
    save_metrics_to_csv, save_preds_targets_to_csv

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = get_timestamp()


# TODO: dokumentacja funkcji """ """

def main():
    print(device)
    epochs = 100
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
    dataset_usage_ratio = 0.01

    # train, val, test subsets proportion f.e. train_ration=0.7 means that it is 70% of the loaded dataset
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    train_loader1, val_loader1, test_loader1 = load_dataset(batch_size, train_ratio, val_ratio, test_ratio,
                                                            train_r_targets1, device, dataset_usage_ratio,
                                                            start_index)
    train_loader2, val_loader2, test_loader2 = load_dataset(batch_size, train_ratio, val_ratio, test_ratio,
                                                            train_r_targets2, device, dataset_usage_ratio,
                                                            start_index)

    # you can choose models: gin, gatv2cn, transformercn, gcn
    model1 = GNNwithMTL.GIN(11, 64, r_targets1).to(device)
    model2 = GNNwithMTL.TransformerCN(11, 64, r_targets2).to(device)

    optimizer = torch.optim.Adam(model2.parameters(), lr=0.0005, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss(reduction='none')

    # print(f'Dla batch = {batch_size}')
    # start = time.time()
    # gnn_train_loss1, gnn_val_loss1 = trainer.train_epochs(epochs, model1,
    #                                                       train_loader1,
    #                                                       val_loader1,
    #                                                       "GNN1.pt",
    #                                                       device, optimizer, loss_fn, r_targets_weights1)
    # end = time.time()
    # print(f"Time = {end - start}")

    print(f'Dla batch = {batch_size}')
    start = time.time()
    gnn_train_loss2, gnn_val_loss2 = trainer.train_epochs(epochs, model2,
                                                          train_loader2,
                                                          val_loader2,
                                                          f"GNN2.pt",
                                                          device, optimizer, loss_fn, timestamp, r_targets_weights2)
    end = time.time()
    print(f"Time = {end - start}")

    trainer.freeze_layers(model2, ['conv'])

    train_ratio_ft = 0.12
    val_ratio_ft = 0.05
    test_ratio_ft = 0.83
    train_r_targets_ft = [11]

    train_loader_ft, val_loader_ft, test_loader_ft = load_dataset(batch_size, train_ratio_ft, val_ratio_ft,
                                                                  test_ratio_ft, train_r_targets_ft, device,
                                                                  0.002, 14080)

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=0.0005,
                                    weight_decay=5e-4)

    gnn_train_loss_ft, gnn_val_loss_ft = trainer.train_epochs(epochs, model2,
                                                              train_loader_ft,
                                                              val_loader_ft,
                                                              f"GNN2FT.pt",
                                                              device, optimizer_ft, loss_fn, timestamp,
                                                              r_targets_weights2)

    # Metrics
    # metrics1, preds1, targets1 = tester.test_gnn(test_loader1, model1, test_r_target1, device)
    # print(f"Test without MTL RMSE: {metrics1['rmse']:.4f}, MAE: {metrics1['mae']:.4f}, R2: {metrics1['r2']:.4f}")
    metrics2, preds2, targets2 = tester.test_gnn(test_loader_ft, model2, test_r_target2, device)
    print(f"Test with MTL RMSE: {metrics2['rmse']:.4f}, MAE: {metrics2['mae']:.4f}, R2: {metrics2['r2']:.4f}")

    # Metrics comparison
    # plot_metric_comparison(metrics1, metrics2, "r2", "experiment without MTL", "experiment with MTL", timestamp)
    # plot_metric_comparison(metrics1, metrics2, "rmse", "experiment without MTL", "experiment with MTL", timestamp)
    # plot_metric_comparison(metrics1, metrics2, "mae", "experiment without MTL", "experiment with MTL", timestamp)

    # Learning curve
    # plot_learning_curve(gin_train_loss1, gin_val_loss1, "gin, experiment without MTL", timestamp)
    plot_learning_curve(gnn_train_loss2, gnn_val_loss2, "transformercn, experiment with MTL", timestamp)
    plot_learning_curve(gnn_train_loss_ft, gnn_val_loss_ft, "transformercn, experiment with MTL FT", timestamp)

    # Parity plot
    # plot_parity_plot(preds1, targets1, "gin, experiment without MTL", timestamp)
    plot_parity_plot(preds2, targets2, "transformercn, experiment with MTL", timestamp)

    # Losses csv
    # save_loss_to_csv(gnn_train_loss1, "gnn_train_loss_without_mtl.csv", timestamp)
    # save_loss_to_csv(gnn_val_loss1, "gnn_val_loss_without_mtl.csv", timestamp)
    save_loss_to_csv(gnn_train_loss2, "gnn_train_loss_with_mtl.csv", timestamp)
    save_loss_to_csv(gnn_val_loss2, "gnn_val_loss_with_mtl.csv", timestamp)
    save_loss_to_csv(gnn_train_loss_ft, "gnn_train_loss_with_mtl_ft.csv", timestamp)
    save_loss_to_csv(gnn_val_loss_ft, "gnn_val_loss_with_mtl_ft.csv", timestamp)

    # Metrics csv
    # save_metrics_to_csv(metrics1, "metrics_without_mtl.csv", timestamp)
    save_metrics_to_csv(metrics2, "metrics_with_mtl_ft.csv", timestamp)

    # PredsTargets csv
    # save_preds_targets_to_csv(preds1, targets1, "preds_targets_without_mtl.csv", timestamp)
    save_preds_targets_to_csv(preds2, targets2, "preds_targets_with_mtl.csv", timestamp)

    return


main()
