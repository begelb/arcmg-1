import numpy as np
import os

level_interval = 20 # change this to change the granularity, try 20
save_dir = 'data'

# dataset = np.loadtxt(os.path.join(save_dir, "dataset_50k.csv"), delimiter=',')
# print('max level set: ', np.max(dataset[:, 2]))
# dataset[:, 3] = (dataset[:, 2] // level_interval) + 1 # floor division + 1

# mul = np.ones_like(dataset[:, 3]) # make a one_vector like the inverted_normalized column
# mul[dataset[:, 4] > 0] = -1 # if the attractor class is 1 or 2, make the one_vector entry negative
# dataset[:, 3] = dataset[:, 3] * mul
# max_level_norm = np.max(dataset[:, 3])
# print('max level norm: ', max_level_norm)
# dataset[:, 3] = dataset[:, 3] / max_level_norm # normalize by the maximum

# dataset = np.loadtxt(os.path.join(save_dir, "dataset_50k.csv"), delimiter=',')
# dataset[:, 4] = (dataset[:, 3] // level_interval) + 1
# mul = np.ones_like(dataset[:, 4])
# mul[dataset[:, 5] > 0] = -1
# dataset[:, 4] = dataset[:, 4] * mul
# max_level_norm = np.max(dataset[:, 4])
# dataset[:, 4] = dataset[:, 4] / max_level_norm

# output_file = os.path.join(save_dir, f'dataset_50k_{level_interval}_corrected.csv')
# np.savetxt(output_file, dataset, delimiter=',')

dataset = np.loadtxt(os.path.join(save_dir, "dataset_50k_original.csv"), delimiter=',')
dataset[:, 3] = (dataset[:, 3] // level_interval) + 1
mul = np.ones_like(dataset[:, 4])
mul[dataset[:, 5] > 0] = -1
dataset[:, 3] = dataset[:, 3] * mul
max_level_norm = np.max(dataset[:, 3])
dataset[:, 4] = dataset[:, 3] / max_level_norm
output_file = os.path.join(save_dir, f'dataset_50k_{level_interval}_corrected.csv')
np.savetxt(output_file, dataset, delimiter=',')
