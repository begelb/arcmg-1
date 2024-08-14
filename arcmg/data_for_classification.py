from torch.utils.data import Dataset
import torch
import os
import numpy as np

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
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        steps_to_attractor = int(data_point[self.d])
        #print('steps to attractor: ', steps_to_attractor)
        attractor = int(data_point[self.d + 1])
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