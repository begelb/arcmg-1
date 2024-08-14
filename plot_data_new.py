import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
import numpy as np

plot_mistakes = True
num_labels = 5
num_attractors = 1
file_name = f'{num_attractors}_att_{num_labels}_classes_model'

# Define the path to the new balanced folder
system = f'pendulum_fixed_{num_attractors}att_{num_labels}_labels'

classifier_path = f'output/{system}/models/simple_classifier.pt'

# Load the classifier
classifier = torch.load(classifier_path)
classifier.eval()

def get_scatter_lists(classifier, subdivisions):
    softmax = torch.nn.Softmax()
# Plot points on a uniform grid and color according to the value of the network
    x_min = -3
    x_max = 3
    y_min = -6
    y_max = 6
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    scatterx = []
    scattery = []
    predictions = []
    for i in range(subdivisions):
        x = x_min + (width/subdivisions) * i
        for j in range(subdivisions):
            y = y_min + (height/subdivisions) * j
            scatterx.append(float(x))
            scattery.append(float(y))

            point = np.zeros(2)
            point[0] += float(x)
            point[1] += float(y)
            point_as_tensor = torch.from_numpy(point).float()
            output = classifier(point_as_tensor)
            probabilities = softmax(output)
            predicted_label = int(torch.argmax(probabilities))
            predictions.append(predicted_label)
    return scatterx, scattery, predictions

def get_label(row, num_labels):
    return min(int(float(row[-1])), num_labels - 1)

def make_figure():
# Create a scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    scatterx, scattery, predictions = get_scatter_lists(classifier, 200)

    cbar_ticks = []
    for i in range(num_labels):
        cbar_ticks.append(i)

    scatter = ax.scatter(scatterx, scattery, marker ='o', s = 6, cmap = 'viridis', c = predictions, alpha = 1)
    cbar = fig.colorbar(scatter, orientation = 'horizontal', fraction=0.05, pad=.13, format="%0.0f", ticks = cbar_ticks)
    plt.title(f'Learned Classes: Pendulum LQR')

    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

make_figure()