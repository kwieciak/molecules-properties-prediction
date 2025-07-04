import time
import warnings
import torch

import utils.utils
from config import timestamp
from data_loader.dataloader import load_dataset
from model import GNNwithMTL, trainer, tester
from utils.utils import save_loss_to_csv, plot_parity_plot, plot_learning_curve, \
    save_metrics_to_csv, save_preds_targets_to_csv, plot_metric_comparison

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO: dokumentacja funkcji """ """

# All hyperparameter names with “1” refer to experiments without MTL
# All hyperparameter names with “2” refer to experiments with MTL

def main():
    print(device)
    epochs = 100
    batch_size = 16
    start_index = 31254

    # qm9 targets:
    # 0 - dipole moment                                   10 - Free energy at 298.15K
    # 1 - isotropic polarizability                        11 - Heat capacity at 298.15K
    # 2 - Highest occupied molecular orbital energy       12 - Atomization energy at 0K
    # 3 - Lowest unoccupied molecular orbital energy      13 - Atomization energy at 298.15K
    # 4 - Gap between 2 and 3                             14 - Atomization enthalpy at 298.15K
    # 5 - Electronic spatial extent                       15 - Atomization free energy at 298.15K
    # 6 - Zero point vibrational energy                   16 - Rotational constant
    # 7 - Internal energy at 0K                           17 - Rotational constant
    # 8 - Internal energy at 298.15K                      18 - Rotational constant
    # 9 - Enthalpy at 298.15K

    # regression targets (tasks) selected to train the model
    train_r_targets1 = [9]
    train_r_targets2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    # regression target (task) selected for model testing
    test_r_target1 = 9
    test_r_target2 = 9

    # merged lists to create the appropriate number of final_linear_layers in the model
    r_targets1 = train_r_targets1 + ([test_r_target1] if test_r_target1 not in train_r_targets1 else [])
    r_targets2 = train_r_targets2 + ([test_r_target2] if test_r_target2 not in train_r_targets2 else [])

    r_targets_weights1 = None
    r_targets_weights2 = None

    # how much of the dataset is taken for the task f.e. dataset_usage_ratio = 0.01 means that it is 1% of the entire qm9 dataset
    dataset_usage_ratio1 = 0.0009
    dataset_usage_ratio2 = 0.75

    # train, val, test subsets proportion f.e. train_ration=0.7 means that it is 70% of the loaded dataset
    train_ratio1 = 0.2
    val_ratio1 = 0.05
    test_ratio1 = 0.75
    train_ratio2 = 0.7
    val_ratio2 = 0.1
    test_ratio2 = 0.2

    train_loader1, val_loader1, test_loader1 = load_dataset(batch_size, train_ratio1, val_ratio1, test_ratio1,
                                                            train_r_targets1, device, dataset_usage_ratio1,
                                                            21241)
    train_loader2, val_loader2, test_loader2 = load_dataset(batch_size, train_ratio2, val_ratio2, test_ratio2,
                                                            train_r_targets2, device, dataset_usage_ratio2,
                                                            start_index)
    print(len(train_loader1.dataset), len(val_loader1.dataset), len(test_loader1.dataset))

    # you can choose models: gin, gatv2cn, transformercn, gcn
    model1 = GNNwithMTL.TransformerCN(11, 64, r_targets1).to(device)
    model2 = GNNwithMTL.TransformerCN(11, 64, r_targets2).to(device)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.0001, weight_decay=0.1)
    loss_fn1 = torch.nn.MSELoss(reduction='none')
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn2 = torch.nn.MSELoss(reduction='none')

    print(f'Dla batch = {batch_size}')
    start = time.time()
    gnn_train_loss1, gnn_val_loss1 = trainer.train_epochs(epochs, model1,
                                                          train_loader1,
                                                          val_loader1,
                                                          "GNN1.pt",
                                                          device, optimizer1, loss_fn1, r_targets_weights1)
    end = time.time()
    print(f"Time = {end - start}")

    print(f'Dla batch = {batch_size}')
    start = time.time()
    gnn_train_loss2, gnn_val_loss2 = trainer.train_epochs(epochs, model2,
                                                          train_loader2,
                                                          val_loader2,
                                                          f"GNN2.pt",
                                                          device, optimizer2, loss_fn2, r_targets_weights2)
    end = time.time()
    print(f"Time = {end - start}")

    trainer.freeze_layers(model2, ['conv'])

    train_ratio_ft = 0.2
    val_ratio_ft = 0.05
    test_ratio_ft = 0.75
    train_r_targets_ft = [9]
    dataset_usage_ratio_ft = 0.0009

    train_loader_ft, val_loader_ft, test_loader_ft = load_dataset(batch_size, train_ratio_ft, val_ratio_ft,
                                                                  test_ratio_ft, train_r_targets_ft, device,
                                                                  dataset_usage_ratio_ft, 21241)

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=0.0001,
                                    weight_decay=0.1)

    print(f'Dla batch = {batch_size}')
    start = time.time()
    gnn_train_loss_ft, gnn_val_loss_ft = trainer.train_epochs(epochs, model2,
                                                              train_loader_ft,
                                                              val_loader_ft,
                                                              f"GNN2FT.pt",
                                                              device, optimizer_ft, loss_fn2,
                                                              r_targets_weights2)
    end = time.time()
    print(f"Time = {end - start}")

    # Metrics
    metrics1, preds1, targets1 = tester.test_gnn(test_loader1, model1, test_r_target1, device)
    print(f"Test without MTL RMSE: {metrics1['rmse']:.4f}, MAE: {metrics1['mae']:.4f}, R2: {metrics1['r2']:.4f}")
    metrics2, preds2, targets2 = tester.test_gnn(test_loader_ft, model2, test_r_target2, device)
    print(f"Test with MTL RMSE: {metrics2['rmse']:.4f}, MAE: {metrics2['mae']:.4f}, R2: {metrics2['r2']:.4f}")
    print(timestamp)

    # Metrics comparison
    plot_metric_comparison(metrics1, metrics2, "r2", "experiment without MTL", "experiment with MTL")
    plot_metric_comparison(metrics1, metrics2, "rmse", "experiment without MTL", "experiment with MTL")
    plot_metric_comparison(metrics1, metrics2, "mae", "experiment without MTL", "experiment with MTL")

    # Learning curve
    plot_learning_curve(gnn_train_loss1, gnn_val_loss1, "gin, experiment without MTL")
    plot_learning_curve(gnn_train_loss2, gnn_val_loss2, "transformercn, experiment with MTL")
    plot_learning_curve(gnn_train_loss_ft, gnn_val_loss_ft, "transformercn, experiment with MTL FT")

    # Parity plot
    plot_parity_plot(preds1, targets1, "gin, experiment without MTL")
    plot_parity_plot(preds2, targets2, "transformercn, experiment with MTL")

    # Losses csv
    save_loss_to_csv(gnn_train_loss1, "gnn_train_loss_without_mtl.csv")
    save_loss_to_csv(gnn_val_loss1, "gnn_val_loss_without_mtl.csv")
    save_loss_to_csv(gnn_train_loss2, "gnn_train_loss_with_mtl.csv")
    save_loss_to_csv(gnn_val_loss2, "gnn_val_loss_with_mtl.csv")
    save_loss_to_csv(gnn_train_loss_ft, "gnn_train_loss_with_mtl_ft.csv")
    save_loss_to_csv(gnn_val_loss_ft, "gnn_val_loss_with_mtl_ft.csv")

    # Metrics csv
    save_metrics_to_csv(metrics1, "metrics_without_mtl.csv")
    save_metrics_to_csv(metrics2, "metrics_with_mtl_ft.csv")

    # PredsTargets csv
    save_preds_targets_to_csv(preds1, targets1, "preds_targets_without_mtl.csv")
    save_preds_targets_to_csv(preds2, targets2, "preds_targets_with_mtl.csv")

    return


main()
