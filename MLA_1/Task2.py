import numpy as np
from Task1 import knn, multi_knn, plot_validation_errors, train_val_splits

# Importing the data
normal_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset.txt"
normal_points = np.loadtxt(normal_file_path).reshape(1877, 784)

low_c_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset-Light-Corruption.txt"
low_c_points = np.loadtxt(low_c_file_path).reshape(1877, 784)

mid_c_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset-Moderate-Corruption.txt"
mid_c_points = np.loadtxt(mid_c_file_path).reshape(1877, 784)

hig_c_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset-Heavy-Corruption.txt"
hig_c_points = np.loadtxt(hig_c_file_path).reshape(1877, 784)

t_labels_path = "MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt"
labels = np.loadtxt(t_labels_path)

# splitting and evaluating error
normal_data, lab = train_val_splits(normal_points, labels, 50, 80, 5)
normal_res = multi_knn(normal_data, lab)

low_c_data, lab = train_val_splits(low_c_points, labels, 50, 80, 5)
low_c_res = multi_knn(low_c_data, lab)

mid_c_data, lab = train_val_splits(mid_c_points, labels, 50, 80, 5)
mid_c_res = multi_knn(mid_c_data, lab)

hig_c_data, lab = train_val_splits(hig_c_points, labels, 50, 80, 5)
hig_c_res = multi_knn(hig_c_data, lab)

# Plotting the results
plot_validation_errors(normal_res)
plot_validation_errors(low_c_res)
plot_validation_errors(mid_c_res)
plot_validation_errors(hig_c_res)