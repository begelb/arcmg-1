import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from arcmg.classifier import Classifier
from arcmg.data_for_classification import DatasetForClassification
from arcmg.config import Config
import yaml
import os

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def save_model(config, model, name):
    model_path = os.path.join(os.getcwd(), config.model_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, os.path.join(model_path, f'{name}.pt'))

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
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            train_loss += loss.item()
            #if (i + 1) % 10 == 0:  # Print every 10 batches

        train_loss /= len(train_loader)
        if epoch % 10 == 0:    
            print(f'Epoch [{epoch + 1}/{config.epochs}], Train Loss: {train_loss:.4f}')

        train_loss = 0.0

        model.eval()
        test_loss = 0.0
        running_accuracy = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                running_accuracy += accuracy(outputs, labels)
                    
            test_loss /= len(test_loader)
            running_accuracy /= len(test_loader)

            if epoch % 10 == 0:
                print(f'Epoch [{epoch + 1}/{config.epochs}], Test Loss: {test_loss:.4f}')
                print('Accuracy: ', f'{running_accuracy:.2f}')
        if running_accuracy > .99:
            break

    save_model(config, model, 'simple_classifier')

    print('Finished Training')

yaml_file = "output/pendulum/config.yaml"

with open(yaml_file, mode="rb") as yaml_reader:
    configuration_file = yaml.safe_load(yaml_reader)

config = Config(configuration_file)

train(config)