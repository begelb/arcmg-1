from arcmg.data_for_supervised_learning import DatasetForClassification, DatasetForRegression
from arcmg.config import Config
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

yaml_file = "output/pendulum_1k/config.yaml"

with open(yaml_file, mode="rb") as yaml_reader:
    configuration_file = yaml.safe_load(yaml_reader)

def get_bins(config):
    data = DatasetForRegression(config, train = True)
    data_loader = DataLoader(data, batch_size=data.__len__(), shuffle=False)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            unique_labels = torch.unique(labels)
          
        unique_label_list = unique_labels.tolist()
        unique_label_list.sort()
        # Here it is assumed that the bins are equally spaced apart, except for the bins closest to zero
        # WLOG, we can compute the difference between the first and second bins to get a general radius
        radius = abs(unique_label_list[1]-unique_label_list[0])

        return unique_label_list, radius
    
def closest_bin(unique_label_list, new_number):
    closest_bin = min(unique_label_list, key=lambda x: abs(x - new_number))
    return closest_bin

def closest_bin_tensor(unique_label_list, predictions):
    unique_label_tensor = torch.tensor(unique_label_list, dtype=predictions.dtype)
    
    # Find the index of the closest bin for each prediction
    diffs = torch.abs(predictions.unsqueeze(-1) - unique_label_tensor)
    indices = diffs.argmin(dim=-1)
    
    # Return the corresponding closest bins
    closest_bins = unique_label_tensor[indices]
    return closest_bins


config = Config(configuration_file)

unique_label_list, radius = get_bins(config)
print('unique label list: ', unique_label_list)
print('closest bin: ', closest_bin(unique_label_list, -0.14))

print('as tensor: ', closest_bin_tensor(unique_label_list, torch.tensor([-0.14, 1, 0.5])))
# print('radius: ', radius)
# my__d = DatasetForRegression(config, train = True)
# print(my__d.data[0])
# dynamics_train_loader = DataLoader(my__d, batch_size=my__d.__len__(), shuffle=False)
# with torch.no_grad():
#     for i, (inputs, labels) in enumerate(dynamics_train_loader):
#         print(inputs)
#         print(labels)
#         unique_labels = torch.unique(labels)
#         print('i: ', i)

#         # Plotting the tensor entries on a number line
#     plt.scatter(unique_labels, torch.zeros(len(unique_labels)), marker='o', color='blue')
#     print(unique_labels)
#     unique_label_list = unique_labels.tolist()

#     for i, label in enumerate(unique_label_list):
#         if i < len(unique_label_list) - 1:
#             print(f'label: {label:.2f}')
#             next_label = unique_label_list[i+1]
#             print(f'next_label: {next_label:.4f}')
#             diff = abs(label - unique_label_list[i+1])
#             print(f'diff: {diff:.4f}')

#     unique_label_list.sort()
#     print(unique_label_list)
#     new_number = -.14
#     closest_number = min(unique_label_list, key=lambda x: abs(x - new_number))
#     print('closest number: ', closest_number)

#     # Adding labels and title
#     plt.yticks([])  # Hide the y-axis ticks

#     # Adding a horizontal line to represent the number line
#     plt.axhline(0, color='black', linewidth=0.5)

#     # Display the plot
#     plt.show()

# print(my__d.d)
# print(my__d.data[0])
# print('-----------------')
# print(my__d.__len__())
# print(my__d.__getitem__(0))

# print((9//20) + 1)
