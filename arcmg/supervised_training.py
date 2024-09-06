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
        self.error_loader = loaders['error_metrics']
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
    
    def get_bins(self):
        data = DatasetForRegression(self.config, train = True)
        data_loader = DataLoader(data, batch_size=data.__len__(), shuffle=False)

        with torch.no_grad():
            for inputs, labels in data_loader:
                unique_labels = torch.unique(labels)
            
            unique_label_list = unique_labels.tolist()
            unique_label_list.sort()
            # Here it is assumed that the bins are equally spaced apart, except for the bins closest to zero
            # WLOG, we can compute the difference between the first and second bins to get a general radius
            radius = abs(unique_label_list[1]-unique_label_list[0])

            return unique_label_list, radius
        
    def closest_bin(self, unique_label_list, new_number):
        closest_bin = min(unique_label_list, key=lambda x: abs(x - new_number))
        return closest_bin
    
    def closest_bin_tensor(self, unique_label_list, predictions):
        unique_label_tensor = torch.tensor(unique_label_list, dtype=predictions.dtype)
        
        # Find the index of the closest bin for each prediction
        diffs = torch.abs(predictions.unsqueeze(-1) - unique_label_tensor)
        indices = diffs.argmin(dim=-1)
        
        # Return the corresponding closest bins
        closest_bins = unique_label_tensor[indices]
        return closest_bins
    
    def accuracy_with_discretization(self, predictions, labels):
        unique_label_list, radius = self.get_bins()
        closest_bins = self.closest_bin_tensor(unique_label_list, predictions)
        matches = closest_bins == labels
        accuracy = matches.float().mean().item()
        return accuracy

    # Here it is assumed that anything labeled positive is in one ROA and anything labeled negative is in another ROA
    # Recall then is the accuracy with respect to this binary classification as positive or negative
    def get_performance_metrics_dict(self, predictions, labels):
        if self.method == 'regression':
            accuracy = self.accuracy_with_discretization(predictions, labels)

            true_pos = torch.sign(labels) > 0
            num_true_pos = true_pos.sum().item()
            num_true_neg = len(labels) - num_true_pos

            predicted_pos = torch.sign(predictions) > 0
            num_predicted_pos = predicted_pos.sum().item()
            num_predicted_neg = len(predictions) - num_predicted_pos

            # For positive labels
            pos_sign_match = (torch.sign(labels) > 0) & (torch.sign(predictions) == torch.sign(labels))
            num_pos_sign_match = pos_sign_match.sum().item()

            # For negative labels
            neg_sign_match = (torch.sign(labels) < 0) & (torch.sign(predictions) == torch.sign(labels))
            num_neg_sign_match = neg_sign_match.sum().item()

            precision_success_att = num_pos_sign_match / num_predicted_pos
            precision_failure_att = num_neg_sign_match / num_predicted_neg
            recall_success_att = num_pos_sign_match / num_true_pos
            recall_failure_att = num_neg_sign_match / num_true_neg

            performance_metrics = {
            'accuracy': accuracy,
            'precision_success_att': precision_success_att,
            'precision_failure_att': precision_failure_att,
            'recall_success_att': recall_success_att,
            'recall_failure_att': recall_failure_att
            }
            
            return performance_metrics

        if self.method == 'classification':
            raise Exception('Recall not implemented')
    
    def get_performance_metrics(self):
        with torch.no_grad():
            for inputs, labels in self.error_loader:
                outputs = self.get_outputs(inputs)
                performance_metrics = self.get_performance_metrics_dict(outputs, labels)
        return performance_metrics
    
    # def write_performance_metrics(self, config):
        

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
    
    def update_performance_metrics(self, performance_metrics, test_loss, train_loss):
        performance_metrics['test_loss'] = float(test_loss)
        performance_metrics['train_loss'] = float(train_loss)
      #  return performance_metrics

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
                    running_train_accuracy += self.accuracy(outputs, labels)

                epoch_train_loss += loss.item()

            
            epoch_train_loss /= len(self.train_loader)
            if self.method == 'classification':
                running_train_accuracy /= len(self.train_loader)

            self.train_losses['loss_total'].append(epoch_train_loss)

            if self.config.verbose:
                if epoch % 10 == 0:    
                    print(f'Epoch [{epoch + 1}/{self.config.epochs}], Train Loss: {epoch_train_loss:.4f}')

            epoch_test_loss = 0.0
            self.model.eval()
            running_test_accuracy = 0.0
            running_precision = 0.0

            with torch.no_grad():
                for i, (inputs, labels) in enumerate(self.test_loader):
                    outputs = self.get_outputs(inputs)

                    loss = criterion(outputs, labels)
                    epoch_test_loss += loss.item()

                    if self.method == 'classification':
                        running_test_accuracy += self.accuracy(outputs, labels)
                    
                    # if self.method == 'regression':
                    #     running_precision += self.recall(outputs, labels)
                        
                epoch_test_loss /= len(self.test_loader)
                self.test_losses['loss_total'].append(epoch_test_loss)

                if self.method == 'classification':
                    running_test_accuracy /= len(self.test_loader)

                # if self.method == 'regression':
                #     running_precision /= len(self.test_loader)

                if self.config.verbose:
                    if epoch % 10 == 0:
                        print(f'Epoch [{epoch + 1}/{self.config.epochs}], Test Loss: {epoch_test_loss:.4f}')
                        if self.method == 'classification':
                            print(f'Train Accuracy: {running_train_accuracy:2f}')
                            print(f'Test Accuracy: {running_test_accuracy:2f}')
                        # if self.method == 'regression':
                        #     print(f'Precision: {running_precision:2f}')

            if self.config.scheduler == 'ReduceLROnPlateau':
                scheduler.step(epoch_test_loss)
            else:
                scheduler.step()

            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    # put recall here 
                    performance_metrics = self.get_performance_metrics()
                    self.update_performance_metrics(performance_metrics, epoch_test_loss, epoch_train_loss)
                    if self.config.verbose:
                        for key, value in performance_metrics.items():
                            print(f"{key}: {value.item():.6f}")
                    return performance_metrics
                
            
            # if self.config.verbose:
            #     print(f"Epoch: {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}")

        performance_metrics = self.get_performance_metrics()
        self.update_performance_metrics(performance_metrics, epoch_test_loss, epoch_train_loss)
        
        if self.config.verbose:
            for key, value in performance_metrics.items():
                print(f"{key}: {value:.2f}")
            return performance_metrics
        #return self.train_losses['loss_total'], self.test_losses['loss_total']
