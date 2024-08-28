import torch 
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from .utils import find_class_order
from .network import PhaseSpaceClassifier
from scipy.stats import norm
from torch.utils.data import DataLoader
from arcmg.classifier import Classifier
from arcmg.data_for_supervised_learning import DatasetForClassification, DatasetForRegression
import csv


# to do: make this just depend on config
# start with dropout after patience turn off dropout
class SupervisedTraining:
    def __init__(self, loaders, config):
        self.config = config
        self.num_labels = self.config.num_labels
        self.lr = self.config.learning_rate
        self.method = self.config.method

        if self.method == 'classification':
            self.model = Classifier(config.input_dimension, config.network_width, config.num_labels)

        if self.method == 'regression':
            self.model = Classifier(config.input_dimension, config.network_width, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loader = loaders['train_dynamics']
        self.test_loader = loaders['test_dynamics']
        self.reset_losses()

    def reset_losses(self):
        self.train_losses = {'loss_total': []}
        self.test_losses = {'loss_total': []}
    
    def save_model(self, name):
        model_path = os.path.join(os.getcwd(),self.config.model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, f'{name}.pt'))

    def load_model(self, name):
        model_path = os.path.join(os.getcwd(),self.config.model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model = torch.load(os.path.join(model_path, f'{name}.pt'))

    def accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)

    # to do: streamline this using class variables
    def write_accuracy_to_csv(self, config, csv_file, train_accuracy, test_accuracy):
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

    def get_dataset(self):
        if self.method == 'classification':
            labeled_dataset = DatasetForClassification(self.config)
        elif self.method == 'regression':
             labeled_dataset = DatasetForRegression(self.config)
        return labeled_dataset
    
    def get_loss_criterion(self):
        if self.method == 'classification':
            criterion = nn.CrossEntropyLoss()
        elif self.method == 'regression':
            criterion = nn.MSELoss()
        return criterion

    def loss_function(self, forward_dict):
        probs_x = forward_dict['probs_x']
        probs_xnext = forward_dict['probs_xnext']
        labels_xnext = forward_dict['labels_xnext']
        labels_x = forward_dict['labels_x']
        return torch.nn.CrossEntropyLoss()(probs_x, self.q_probability_vector(probs_xnext,labels_xnext))

    def get_optimizer(self, list_parameters):
        if self.config.optimizer == 'Adam':
            return torch.optim.Adam(list_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'SGD':
            return torch.optim.SGD(list_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'Adagrad':
            return torch.optim.Adagrad(list_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'AdamW':
            return torch.optim.AdamW(list_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'RMSprop':
            return torch.optim.RMSprop(list_parameters, lr=self.config.learning_rate) # not compatible with cyclic LR
        elif self.config.optimizer == 'Adadelta':
            return torch.optim.Adadelta(list_parameters, lr=self.config.learning_rate) # not compatible with cyclic LR

    def get_scheduler(self, optimizer):
        if self.config.scheduler == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.config.learning_rate, max_lr=0.1)
        elif self.config.scheduler == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.config.patience, verbose=self.config.verbose)

    def get_outputs(self, inputs):
        if self.method == 'classification':
            outputs = self.model(inputs)
        elif self.method == 'regression':
            outputs = self.model(inputs).squeeze()
        return outputs

    def train(self):
        epochs = self.config.epochs
        patience = self.config.patience
        list_parameters = list(self.model.parameters())
        optimizer = self.get_optimizer(list_parameters)
        scheduler = self.get_scheduler(optimizer)

        for epoch in tqdm(range(epochs)):
            epoch_train_loss = 0.0
            running_train_accuracy = 0.0
            self.model.train()

            # loop over the batches in the train_loadaer 
            for i, (inputs, labels) in enumerate(self.train_loader):
                
                optimizer.zero_grad()

                outputs = self.get_outputs(inputs)

                criterion = self.get_loss_criterion()
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # Train accuracy
                if self.method == 'classification':
                    running_train_accuracy += accuracy(outputs, labels)

                epoch_train_loss += loss.item()

            
            epoch_train_loss /= len(self.train_loader)
            if self.method == 'classification':
                running_train_accuracy /= len(self.train_loader)

            self.train_losses['loss_total'].append(epoch_train_loss)

            if self.config.verbose:
                if epoch % 10 == 0:    
                    print(f'Epoch [{epoch + 1}/{self.config.epochs}], Train Loss: {self.train_loss:.4f}')

            epoch_test_loss = 0.0
            self.model.eval()
            running_test_accuracy = 0.0

            with torch.no_grad():
                for i, (inputs, labels) in enumerate(self.test_loader):
                    outputs = self.get_outputs(inputs)

                    loss = criterion(outputs, labels)
                    epoch_test_loss += loss.item()

                    if self.method == 'classification':
                        running_test_accuracy += self.accuracy(outputs, labels)
                        
                epoch_test_loss /= len(self.test_loader)
                self.test_losses['loss_total'].append(epoch_test_loss)

                if self.method == 'classification':
                    running_test_accuracy /= len(self.train_loader)

                if self.config.verbose:
                    if epoch % 10 == 0:
                        print(f'Epoch [{epoch + 1}/{self.config.epochs}], Test Loss: {epoch_test_loss:.4f}')
                        if self.method == 'classification':
                            print(f'Train Accuracy: {running_train_accuracy:2f}')
                            print(f'Test Accuracy: {running_test_accuracy:2f}')

            if self.config.scheduler == 'ReduceLROnPlateau':
                scheduler.step(epoch_test_loss)
            else:
                scheduler.step()

            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    return epoch_test_loss
            
            # if self.config.verbose:
            #     print(f"Epoch: {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}")

        #return self.train_losses['loss_total'], self.test_losses['loss_total']
