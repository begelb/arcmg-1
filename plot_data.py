import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch

plot_mistakes = True
num_labels = 9
file_name = f'{num_labels}_classes_model'

# Define the path to the new balanced folder
system = f'pendulum_step5'
#system = 'balanced_pendulum_three_labels'
folder= f'data/{system}'

classifier_path = f'output/{system}/models/simple_classifier_{num_labels}labels.pt'

def get_label(row, num_labels):
    return min(int(float(row[-1])), num_labels - 1)

# Load the classifier
classifier = torch.load(classifier_path)
classifier.eval()

# Initialize lists to store data points and labels
features_x = []
features_y = []
labels = []
unique_labels = set()

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path):
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for index, row in enumerate(reader):
                label = get_label(row, num_labels)
                features_x.append(float(row[0]))
                features_y.append(float(row[1]))
                labels.append(label)
                unique_labels.add(label)

print('Unique labels:', unique_labels)
# Convert data to tensor
data = torch.tensor(list(zip(features_x, features_y)))
predictions = []

softmax = torch.nn.Softmax(dim=1)
if num_labels == 5:
    colors = ['#440154FF', '#404688FF', '#2A788EFF', '#7AD151FF', '#FDE725FF']
    markers = ['o', 's', 'D', '^', 'P']
if num_labels == 7:
    colors = ['#440154FF', '#443A83FF', '#31688EFF', '#1F9A8AFF', '#47C16EFF', '#8FD744FF', '#FDE725FF']
    markers = ['o', 's', 'D', '^', 'P', 'X', 'v']
if num_labels == 9:
    colors = ['#440154FF', '#443A83FF', '#404688FF', '#31688EFF', '#287C8EFF', '#1F9A8AFF', '#47C16EFF', '#8FD744FF', '#FDE725FF']
    markers = ['o', 's', 'D', '^', 'P', 'X', 'v', '>', '<']
#47C16EFF#5DC863FF#75D054FF
if plot_mistakes:
    # Get predictions
    with torch.no_grad():
        for point in data:
            point = point.unsqueeze(0)
            output = classifier(point)
            probabilities = softmax(output)
            predicted_label = int(torch.argmax(output))
            predictions.append(predicted_label)
    

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    
    for i, (x, y) in enumerate(zip(features_x, features_y)):
        if predictions[i] == labels[i]:
            color = colors[labels[i]]
        else:
            color = 'black'

        plt.scatter(x, y, color=color, marker=markers[labels[i]])


    unique_labels = sorted(set(labels))
    for label in unique_labels:
        plt.scatter([], [], color=colors[label], marker=markers[label], label=f'Label {label}')

else:
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        label_mask = [l == label for l in labels]
        plt.scatter(
            [features_x[i] for i in range(len(features_x)) if label_mask[i]],
            [features_y[i] for i in range(len(features_y)) if label_mask[i]],
            color=colors[label],
            marker=markers[label],
            label=f'Label {label}'
        )

# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
if plot_mistakes:
    plt.title(f'Learned Classes: Pendulum LQR, 1K Trajectories, Step = 5 (in processing script), Mistakes in Black')
else:
    plt.title(f'Labeled Data: Pendulum LQR, 1K Trajectories, Step = 5 (in processing script)')
#plt.legend(fontsize = 15, loc = 'upper right', framealpha = 1)
plt.grid(True)
plt.savefig(file_name)
plt.show()