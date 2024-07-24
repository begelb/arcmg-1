import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from arcmg.classifier import Classifier
from arcmg.data_for_classification import DatasetForClassification
from arcmg.config import Config
import yaml

def train(config):

    data = DatasetForClassification(config)

    # Dataset and DataLoader
    train_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)

    # Model, Loss, and Optimizer

    model = Classifier(config.input_dimension, config.network_width, config.num_labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
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
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch + 1}/{config.epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    print('Finished Training')

yaml_file = "output/pendulum/config.yaml"

with open(yaml_file, mode="rb") as yaml_reader:
    configuration_file = yaml.safe_load(yaml_reader)

config = Config(configuration_file)

train(config)