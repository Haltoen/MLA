import numpy as np
import matplotlib.pyplot as plt

# loading the data
t_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset.txt"
points = np.loadtxt(t_file_path).reshape(1877, 784)
t_labels_path = "MNIST-5-6-Subset/MNIST-5-6-Subset-Labels.txt"
labels = np.loadtxt(t_labels_path)

def train_val_splits(data, labels, m: int, n: int, a: int) -> ([np.array] , [np.array]):
    """
    Makes the traioning and valiadation splits. 

    Parameters:
    - data (): d x m matrix of points
    - labels (): Vector of labels
    - m (int): Amount of samples in the test_set
    - n (int): Amount of samples in each validation set
    - a (int): Amount of validation sets

    Returns:
    - (list np.array, list np.array) Training and vadition splits,
        The first index contains lists of data points, and the second contains labels.
        index 0 in these lists is for training, and 1 to a is for the validation sets
    """

    # checking if input params are valid
    if a <= 0:
        raise ValueError("Number of validation sets (a) must be greater than 0.")
    if m + n * a >= data.shape[1]:
        raise ValueError("Total number of validation samples exceeds the data size.")

    # Initialize lists to store training and validation splits
    data_lst = []
    label_lst = []

    # add the training sets 
    data_lst.append(data[:m, :])
    label_lst.append(labels[:m])

    print(data.shape)
    print (labels.shape)

    # add the a validation sets
    for i in range(a):
        # Calculate the starting and ending indices for the current validation set
        start_idx = i * n + m
        end_idx = (i + 1) * n + m

        # Extract the validation set
        val_data = data[start_idx:end_idx, : ]
        val_labels = labels[start_idx:end_idx]

        print (val_data.shape)
        print (val_labels.shape)


        # Append the training and validation sets to the respective lists
        data_lst.append(val_data)
        label_lst.append(val_labels)

    return data_lst, label_lst

def knn(training_points: np.ndarray, training_labels: np.ndarray, test_points: np.ndarray, test_labels: np.ndarray):
    """
    Perform k-NN classification for given test points and a single k value.

    Parameters:
    - training_points (numpy.ndarray): d x m matrix of training points.
    - training_labels (numpy.ndarray): Vector of corresponding training labels.
    - test_points (numpy.ndarray): d x n matrix of test points.
    - test_labels (numpy.ndarray): Vector of corresponding training labels.
    - k (int): Number of nearest neighbors to consider.

    Returns:
    - predicted_labels (numpy.ndarray): Predicted labels for test points.
    """
    # Convert labels {5,6} to {-1,1}
    training_labels[training_labels == 5] = -1
    training_labels[training_labels == 6] = 1
    test_labels[test_labels == 5] = -1
    test_labels[test_labels == 6] = 1

    # Reshape the data for broadcasting
    training_points = training_points.T # Transpose to make it (m, d)
    test_points = test_points.T  # Transpose to make it (n, d)

    print ("points shape", training_points.shape, test_points.shape)
    print ("labels shape", training_labels.shape, test_labels.shape)

    # Distances between all test and training points
    distances = np.linalg.norm(training_points[:, :, np.newaxis] - test_points[:, np.newaxis, :], axis=0)

    m = len(training_labels)
    n = len(test_labels)
    errors = np.zeros(m, dtype=float)

    # Calculate errors for k = 1, 2, ..., m
    for k in range(1, m + 1):
        # Get the indices of the k-nearest neighbors for each test point
        k_nearest_indices = np.argsort(distances, axis=0)[:k, :]

        # Get the labels of the k-nearest neighbors for each test point
        k_nearest_labels = training_labels[k_nearest_indices]

        # Predict the labels for each test point using a majority vote
        predicted_labels = np.sign(np.sum(k_nearest_labels, axis=0))

        # Calculate the average error for each k
        errors[k - 1] = round((np.sum(predicted_labels != test_labels, axis=0)/m),2)

    return errors


def multi_knn(data, labels):
    """
    Evaluate knn for all validations sets outputted by train_val_split.
    """
    train_data = data[0]
    train_labels = labels[0]

    m = len(train_data)
    n = len(data) - 1

    results = np.zeros((m,n), dtype=float)

    for i in range(1, n+1):
        test_data = data[i]
        test_labels = labels[i]

        error = knn(train_data, train_labels, test_data, test_labels)
        results[:, i-1] = error
    
    return results # use this as input for error plots

def plot_validation_errors(Results):
    """
    Plot validation error rates for different validation sets as a function of K.

    Parameters:
    - Results (numpy.ndarray): Array containing validation error rates for each validation set and K.
    """

    # Get the number of validation sets and values of K
    num_K_values, num_validation_sets  = Results.shape
    print(num_K_values)
    K_values = range(1, num_K_values + 1)

    # Plot the validation error for each validation set
    for i in range(num_validation_sets):
        plt.plot(K_values, Results[:, i], label=f'Validation Set {i + 1}')

    plt.xlabel('K')
    plt.ylabel('Validation Error Rate')
    plt.title('Validation Error Rate vs. K for Different Validation Sets')
    plt.legend()
    plt.show()



def plot_variance_of_validation_errors(Results):
    """
    Plot variance of validation error rates for different validation sets as a function of K.

    Parameters:
    - Results (numpy.ndarray): Array containing validation error rates for each validation set and K.
    """

    def calculate_variance(data):
        # Calculate the mean (average)
        mean = sum(data) / len(data)

        # Calculate the squared differences from the mean
        squared_diff = [(x - mean) ** 2 for x in data]
        return squared_diff
    
    # Get the number of validation sets and values of K
    num_K_values, num_validation_sets = Results.shape
    K_values = range(1, num_K_values + 1)

    # Plot the variances as a function of K for each validation set
    for i in range(num_validation_sets):
        validation_set_variances = calculate_variance(Results[:, i])
        plt.plot(K_values, validation_set_variances, label=f'Variance of Validation Errors (Set {i + 1})')

    plt.xlabel('K')
    plt.ylabel('Variance of Validation Error Rates')
    plt.title('Variance of Validation Error Rates vs. K for Different Validation Sets')
    plt.legend()
    plt.show()

data, labels = train_val_splits(points, labels, 50, 80, 5)

res = multi_knn(data, labels)

plot_variance_of_validation_errors(res)


