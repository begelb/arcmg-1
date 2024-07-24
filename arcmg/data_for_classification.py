from torch.utils.data import Dataset
import torch
import os
import numpy as np

class DatasetForClassification(Dataset):
    def __init__(self, config):
        
        self.d = config.input_dimension
        self.num_labels = config.num_labels
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
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        labeled_point_pair = [data_point[:self.d],
                              self.truncate_labels(int(data_point[self.d]))]

        return labeled_point_pair