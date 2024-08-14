import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from arcmg.classifier import Classifier
from arcmg.data_for_classification import DatasetForClassification
from arcmg.config import Config
import yaml
import os
import csv
import argparse

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def save_model(config, model, name):
    model_path = os.path.join(os.getcwd(), config.model_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, os.path.join(model_path, f'{name}.pt'))

def write_accuracy_to_csv(config, csv_file, train_accuracy, test_accuracy):
    # Check if the file exists
    file_exists = os.path.isfile(csv_file)
    
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writerow(["num_attractors", "num_labels", "train_accuracy", "test_accuracy"])
        
        # Write the accuracy values
        writer.writerow([config.num_attractors, config.num_labels, train_accuracy, test_accuracy])

def train(config):

    labeled_dataset = DatasetForClassification(config)

    train_size = int(0.8*len(labeled_dataset))
    test_size = len(labeled_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(labeled_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Model, Loss, and Optimizer

    model = Classifier(config.input_dimension, config.network_width, config.num_labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        running_train_accuracy = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Train accuracy
            running_train_accuracy += accuracy(outputs, labels)

            # Print statistics
            train_loss += loss.item()
            #if (i + 1) % 10 == 0:  # Print every 10 batches

        train_loss /= len(train_loader)
        running_train_accuracy /= len(train_loader)

        if epoch % 10 == 0:    
            print(f'Epoch [{epoch + 1}/{config.epochs}], Train Loss: {train_loss:.4f}')

        train_loss = 0.0

        model.eval()
        test_loss = 0.0
        running_test_accuracy = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                running_test_accuracy += accuracy(outputs, labels)
                    
            test_loss /= len(test_loader)
            running_test_accuracy /= len(test_loader)

            if epoch % 10 == 0:
                print(f'Epoch [{epoch + 1}/{config.epochs}], Test Loss: {test_loss:.4f}')
                print(f'Train Accuracy: {running_train_accuracy:2f}')
                print(f'Test Accuracy: {running_test_accuracy:2f}')
        #if running_test_accuracy > .99:
        #    break

    save_model(config, model, 'simple_classifier')
    results_file = config.output_dir + 'accuracy.csv'
    write_accuracy_to_csv(config, results_file, running_train_accuracy, running_test_accuracy)
    #print('Finished Training')

def main(args, yaml_file):
    yaml_file_path = args.config_dir

    with open(os.path.join(yaml_file_path, yaml_file), mode="rb") as yaml_reader:
        configuration_file = yaml.unsafe_load(yaml_reader)

    config = Config(configuration_file)

    config.output_dir = os.path.join(os.getcwd(),config.output_dir)

    train(config)

if __name__ == "__main__":

# 
    yaml_file_path = os.getcwd() + "/output/pendulum_fixed"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default=yaml_file_path)
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default="config.yaml")
    parser.add_argument('--transfer_learning',help='Config file inside config_dir',action='store_true')
    #  parser.add_argument('--verbose',help='Print training output',action='store_true')

    args = parser.parse_args()
    
    main(args, args.config) 