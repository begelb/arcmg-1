from arcmg.supervised_training import SupervisedTraining
from arcmg.config import Config
from arcmg.data_for_supervised_learning import DatasetForClassification, DatasetForRegression
from arcmg.plot import plot_classes, plot_loss, plot_classes_2D
from arcmg.utils import grid_points2d
from torch.utils.data import DataLoader
import argparse
import csv
import torch
import yaml
import os
from torch import nn
import pandas as pd

torch.autograd.set_detect_anomaly(True)

num_points_in_mesh = 1000

def main(args, yaml_file):
    ###
    # yaml_file = "config/rampfn.yaml"
    yaml_file_path = args.config_dir

    with open(os.path.join(yaml_file_path, yaml_file), mode="rb") as yaml_reader:
        configuration_file = yaml.unsafe_load(yaml_reader)

    config = Config(configuration_file)
    # config.check_types()

    config.output_dir = os.path.join(os.getcwd(),config.output_dir)

    if config.method == 'classification':
        dynamics_dataset = DatasetForClassification(config)
    elif config.method == 'regression':
        dynamics_dataset = DatasetForRegression(config, train = True)

    dataset_for_performance_metrics = DatasetForRegression(config, train = False)

    dynamics_train_size = int(0.8*len(dynamics_dataset))
    dynamics_test_size = len(dynamics_dataset) - dynamics_train_size
    dynamics_train_dataset, dynamics_test_dataset = torch.utils.data.random_split(dynamics_dataset, [dynamics_train_size, dynamics_test_size], generator=torch.Generator().manual_seed(config.seed_data_split))
    dynamics_train_loader = DataLoader(dynamics_train_dataset, batch_size=config.batch_size, shuffle=True)
    dynamics_test_loader = DataLoader(dynamics_test_dataset, batch_size=config.batch_size, shuffle=True)
    performance_metrics_loader = DataLoader(dataset_for_performance_metrics, batch_size=dataset_for_performance_metrics.__len__(), shuffle=False)

    if config.verbose:
        print("Train size: ", len(dynamics_train_dataset))
        print("Test size: ", len(dynamics_test_dataset))

    loaders = {
        'train_dynamics': dynamics_train_loader,
        'test_dynamics': dynamics_test_loader,
        'performance_metrics': performance_metrics_loader
    }

    # save performance metrices to a result file
    with open(config.output_dir + 'result.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Set random seed for the remaining torch processes
        torch.manual_seed(config.seed_nn_init)
        trainer = SupervisedTraining(loaders, config)

        print(trainer.model)

        if not args.only_plot:

            performance_metrics = trainer.train()
            if config.method == 'classification':
                trainer.save_model('sup_classifier')
            elif config.method == 'regression':
                trainer.save_model('regression')

            # Write the header (column names)
            writer.writerow(performance_metrics.keys())
            
            # Write the first row (values)
            writer.writerow(performance_metrics.values())

       # trainer.load_model('classifier')

        # plot_classes(trainer.model, config)  
        # plot_classes_2D(trainer.model, config)
        #heatmap(trainer.model, config)          

        train_losses = trainer.train_losses['loss_total']
        test_losses = trainer.test_losses['loss_total']
        plot_loss(config, train_losses, test_losses)

        # class_set = find_classes(trainer.model, config, num_points_in_mesh)
        # num_classes_found = len(class_set)

        # writer.writerow(["class_set", "num_classes_found", "is_bistable", "final_train_loss", "final_test_loss"])
        # writer.writerow([class_set, num_classes_found, is_bistable(class_set), train_losses[-1], test_losses[-1]])

    points = grid_points2d(config, 10)
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model(points)
    grid_predictions = torch.cat([points, outputs], dim=1)
    grid_predictions_np = grid_predictions.numpy()
    df = pd.DataFrame(grid_predictions_np, columns=['x', 'y', 'output'])
    print(df)
    df.to_csv(config.output_dir + 'grid_predictions.csv', index=False)

if __name__ == "__main__":

#  This is actually a folder not a path? 
    yaml_file_path = os.getcwd() + "/output/pendulum_1k"

    only_plot = False

    parser = argparse.ArgumentParser()
    #  parser.add_argument('--job_index',help='Job index',type=int,default=0)
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default=yaml_file_path)
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default="config.yaml")
    parser.add_argument('--transfer_learning',help='Config file inside config_dir',action='store_true')
    parser.add_argument('--only_plot',help='Load model and plot',type=bool,default=only_plot)
    #  parser.add_argument('--verbose',help='Print training output',action='store_true')

    args = parser.parse_args()

    # args.transfer_learning = True

    main(args, args.config) 