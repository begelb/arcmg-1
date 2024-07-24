import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch

plot_mistakes = False
file_name = 'five_classes_data'

def get_label(row, num_labels):
    return min(int(float(row[-1])), num_labels - 1)

# Define the path to the new balanced folder
system = 'pendulum'
#system = 'balanced_pendulum_three_labels'
folder= f'data/{system}'

classifier_path = f'output/{system}/models/classifier.pt'

num_labels = 5

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

# Convert data to tensor
data = torch.tensor(list(zip(features_x, features_y)))
predictions = []

if plot_mistakes:
    # Get predictions
    with torch.no_grad():
        for point in data:
            point = point.unsqueeze(0)
            output = classifier.vector_of_probabilities(point)
            predicted_label = int(torch.argmax(output))
            predictions.append(predicted_label)
        # point = data.unsqueeze(0)
        # output = classifier.vector_of_probabilities(point)
        # predictions = int(torch.argmax(output))
        
    # Define colors for each label
    colors = ['blue', 'green', 'yellow', 'purple', 'gray']  # Adjust or add more colors as needed

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(zip(features_x, features_y)):
        if predictions[i] == labels[i]:
            color = colors[labels[i]]
        else:
            print('prediction: ', predictions[i])
            print('labels: ', labels[i])#
            color = 'red'
        plt.scatter(x, y, color=color)

else:
    colors = ['blue', 'green', 'yellow', 'purple', 'gray']
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        label_mask = [l == label for l in labels]
        plt.scatter(
            [features_x[i] for i in range(len(features_x)) if label_mask[i]],
            [features_y[i] for i in range(len(features_y)) if label_mask[i]],
            label=f'Label {label}'
        )

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Labeled Data Scatter Plot')
plt.legend()
plt.grid(True)
plt.savefig(file_name)
plt.show()