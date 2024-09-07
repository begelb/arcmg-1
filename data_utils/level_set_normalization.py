import numpy as np
import os

level_interval = 20 # change this to change the granularity, try 20
save_dir = 'data'

dataset = np.loadtxt(os.path.join(save_dir, "dataset_50k_original.csv"), delimiter=',')
dataset[:, 3] = (dataset[:, 3] // level_interval) + 1
mul = np.ones_like(dataset[:, 4])
mul[dataset[:, 5] > 0] = -1
dataset[:, 3] = dataset[:, 3] * mul
max_level_norm = np.max(dataset[:, 3])
dataset[:, 4] = dataset[:, 3] / max_level_norm
output_file = os.path.join(save_dir, f'dataset_50k_{level_interval}_corrected.csv')
np.savetxt(output_file, dataset, delimiter=',')
