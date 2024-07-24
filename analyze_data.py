import csv
import os
import shutil

original_folder_path = 'data/pendulum'
new_folder_path = 'data/balanced_pendulum_three_labels'

os.makedirs(new_folder_path, exist_ok=True)

num_labels = 3

label_file_dict = {i: [] for i in range(num_labels)}

def get_label(row, num_labels):
    return min(int(float(row[-1])), num_labels - 1)


def check_distribution(folder):
    num_label_dict = {i: 0 for i in range(num_labels)}
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            with open(file_path, newline='') as f:
                reader = csv.reader(f)
                try:
                    next(reader)  # Skip the first row
                except:
                    os.remove(file_path)
                for index, row in enumerate(reader):
                    label = get_label(row, num_labels)
                    num_label_dict[label] += 1
    for i in range(num_labels):
        print(f'Label {i}: {num_label_dict[i]}')
    return num_label_dict
                    

num_label_dict_original = check_distribution(original_folder_path)
min_count = min(num_label_dict_original.values())

num_copied_labels_dict = {i: 0 for i in range(num_labels)}

for j, filename in enumerate(os.listdir(original_folder_path)):
    old_file = os.path.join(original_folder_path, filename)
    new_file = os.path.join(new_folder_path, filename)
    with open(new_file, mode='w', newline='') as w:
        writer = csv.writer(w)
        
        with open(old_file, newline='') as f:
            reader = csv.reader(f)
            row_list = list(reader)
            for j, row in enumerate(row_list[:-1]):
                label = get_label(row_list[j+1], num_labels)
                if num_copied_labels_dict[label] < min_count:
                    num_copied_labels_dict[label] += 1
                    if j == 0:
                        writer.writerow(row_list[j])
                        writer.writerow(row_list[j+1])
                    else:
                        writer.writerow(row_list[j+1])

print('--------------------------')
check_distribution(new_folder_path)
print('-----')
print(num_copied_labels_dict)

# num_labels = 5
# num_label_dict = dict()
# num_label_dict = {i: 0 for i in range(num_labels)}
# total_amt_data = 0

# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#     if os.path.isfile(file_path):
#         with open(file_path, newline='') as f:
#             reader = csv.reader(f)
#             next(reader) # skip first row
#             for row in reader:
#                 total_amt_data += 1
#                 label = get_label(row)
#                 num_label_dict[label] += 1

# for i in range(num_labels):
#     ratio = num_label_dict[i]/total_amt_data
#     print(f'Label {i}: {ratio}')
    