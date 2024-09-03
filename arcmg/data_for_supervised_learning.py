from torch.utils.data import Dataset
import torch
import os
import numpy as np

class DatasetForRegression(Dataset):
    def __init__(self, config, train):
        self.d = config.input_dimension
        X=[]

        if train:
            for f in os.listdir(config.data_file):
                # load data into a numpy array
                data = np.loadtxt(os.path.join(config.data_file, f), delimiter=',')
                X.append(data)
        else:
            for f in os.listdir(config.data_file_test):
                # load data into a numpy array
                data = np.loadtxt(os.path.join(config.data_file_test, f), delimiter=',')
                X.append(data)
            
        self.X = np.vstack(X)
            
        self.data = torch.from_numpy(self.X).float()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_point = self.data[idx]

        labeled_point_pair = [data_point[:self.d],
                              data_point[4]]

        return labeled_point_pair

class DatasetForClassification(Dataset):
    def __init__(self, config):
        
        self.d = config.input_dimension
        self.num_labels = config.num_labels
        self.num_attractors = config.num_attractors
        self.distinguish_attractors = config.distinguish_attractors
        X=[]

        for f in os.listdir(config.data_file):
            # load data into a numpy array
            data = np.loadtxt(os.path.join(config.data_file, f), delimiter=',')
            X.append(data)
            
        self.X = np.vstack(X)
            
        self.data = torch.from_numpy(self.X).float()

    def __len__(self):
        return len(self.data)
    
    def truncate_labels(self, label):
        return min(label, self.num_labels - 1)
    
    ''' attractor is expected to be an integer such as 0, 1, 2, ...'''
    def distinguish_level_sets(self, steps_to_attractor, attractor):
        # attractor is expected to be an integer such as 0, 1, 2, ...
        return steps_to_attractor * self.num_attractors + attractor
    
    def determine_attractor(self, tmp_attractor):
        if self.num_attractors == 1:
            return 0
        if self.num_attractors == 2:
            if tmp_attractor == 1 or tmp_attractor == 2:
                return 1
            else:
                return 0
        if self.num_attractors == 3:
            return tmp_attractor
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        steps_to_attractor = int(data_point[self.d])
        #print('steps to attractor: ', steps_to_attractor)
        tmp_attractor = int(data_point[self.d + 1])
        attractor = self.determine_attractor(tmp_attractor)
        
        #print('attractor: ', attractor)
        if self.distinguish_attractors:
            level_set = self.distinguish_level_sets(steps_to_attractor, attractor)
            #print('level set: ', level_set)
            level_set_truncated = self.truncate_labels(level_set)
            #print('truncated: ', level_set_truncated)
        else:
            level_set_truncated = self.truncate_labels(steps_to_attractor)

        labeled_point_pair = [data_point[:self.d],
                              level_set_truncated]

        return labeled_point_pair