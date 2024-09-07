import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
import numpy as np

plot_mistakes = False
num_labels = 20
num_attractors = 2
#file_name = f'{num_attractors}_att_{num_labels}_classes_model'
file_name = 'regression_model'

# Define the path to the new balanced folder
#system = f'pendulum_fixed_{num_attractors}att_{num_labels}_labels'
system = 'pendulum_1k'
data_folder = f'experiments/data/{system}'

classifier_path = f'output/{system}/models/regression.pt'

# Load the classifier
# classifier = torch.load(classifier_path)
# classifier.eval()

def get_data(data_folder):
    features_x = []
    features_y = []
    labels = []
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        if os.path.isfile(file_path):
            with open(file_path, newline='') as f:
                reader = csv.reader(f)
                next(reader)
                for index, row in enumerate(reader):
                    if index < 200000:
                        label = float(row[4])
                        features_x.append(float(row[0]))
                        features_y.append(float(row[1]))
                        labels.append(label)
    return features_x, features_y, labels


def get_scatter_lists(classifier, subdivisions, classification):
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

            with torch.no_grad():
                point_as_tensor = torch.from_numpy(point).float()

                output = classifier(point_as_tensor)
                if classification:
                    probabilities = softmax(output)
                    predicted_label = int(torch.argmax(probabilities))
                    predictions.append(predicted_label)
                else:
                    predictions.append(output)
    return scatterx, scattery, predictions

def get_label(row, num_labels):
    return min(int(float(row[-1])), num_labels - 1)


def make_figure(classification, plotting_data):
# Create a scatter plot
   # fig = plt.figure(figsize=(10, 6))
    fig = plt.figure(figsize=(8,10))
    ax = fig.add_subplot(111)

    if plotting_data:
        scatterx, scattery, predictions = get_data(data_folder)
    else:
        classifier_path = f'output/{system}/models/regression.pt'

        # Load the classifier
        classifier = torch.load(classifier_path)
        classifier.eval()
        scatterx, scattery, predictions = get_scatter_lists(classifier, 200, classification)

    if classification:
        cbar_ticks = []
        for i in range(num_labels):
            cbar_ticks.append(i)
    else:
        cbar_ticks = np.linspace(-1, 1, 11).tolist()
        print(cbar_ticks)

    scatter = ax.scatter(scatterx, scattery, marker ='o', s = 6, cmap = 'rainbow', c = predictions, alpha = 1)
    cbar = fig.colorbar(scatter, orientation = 'horizontal', fraction=0.05, pad=.13, format="%0.2f")
    
    if plotting_data:
        plt.title(f'Data: Pendulum LQR')
    else:
        plt.title(f'Pendulum LQR 1K Trajectories, Regression')

    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

make_figure(classification = False, plotting_data = False)
print('Done')