import numpy as np

# Train a perceptron model with the given parameters#


def perceptron_train(trainset_values, trainset_labels, learning_rate, maxit):
    n_samples, n_features = trainset_values.shape
    w = np.zeros(n_features)
    b = 0

    for _ in range(maxit):
        errors = 0
        for i in range(n_samples):
            # calculate predicted label
            predict = np.sign(np.dot(trainset_values[i], w) + b)
            # update weights if predicted label doesn't match true label
            if predict != trainset_labels[i]:
                w += learning_rate * trainset_labels[i] * trainset_values[i]
                b += learning_rate * trainset_labels[i]
                errors += 1
        # stop if no errors were made during an epoch
        if errors == 0:
            break

    return w, b


# Predict labels for data with trained perceptron model


def perceptron_predict(trainset_values, w, b):
    prediction = np.sign(np.dot(trainset_values, w) + b)
    return prediction


def create_confusion_matrix(prediction, original):
    labels = np.unique(original)
    amount = len(labels)

    confusion_matrix = np.zeros((amount, amount))

    # Fill in the confusion matrix with the number of predictions for each pair of true and predicted labels
    for i in range(amount):
        for j in range(amount):
            confusion_matrix[i, j] = np.sum((original == labels[i]) & (
                prediction == labels[j]))

    return confusion_matrix.astype(int)


def calculate_accuracy(prediction, original):
    return np.mean(prediction == original)


list_folders = [
    {
        'train_file': 'data//spam//spam.data',
        'test_file': 'data//spam//spam.data',
    },
    {
        'train_file': 'data//leukemia//ALLAML.trn',
        'test_file': 'data//leukemia//ALLAML.tst',
    },
    {
        'train_file': 'data//ovarian//ovarian.data',
        'test_file': 'data//ovarian//ovarian.data',
    },
]

learning_rate = 0.05
maxit = 490

# Loop over all folders
for folder in list_folders:
    print('Learning rate:', learning_rate)
    print('Maxit:', maxit)
    print('Folder:', folder['test_file'].split('//')[1])
    # Load training and testing data from files
    try:
        train_data = np.loadtxt(
            folder['train_file'], delimiter=',', dtype=float)
        test_data = np.loadtxt(
            folder['test_file'], delimiter=',', dtype=float)
    except:
        train_data = np.loadtxt(
            folder['train_file'], delimiter=' ', dtype=float)
        test_data = np.loadtxt(
            folder['test_file'], delimiter=' ', dtype=float)

    trainset_values, trainset_labels = train_data[:,
                                                  :-1], train_data[:, -1].astype(int)
    testset_values, testset_labels = test_data[:,
                                               :-1], test_data[:, -1].astype(int)

    # train perceptron model
    w, eta = perceptron_train(
        trainset_values, trainset_labels, learning_rate, maxit)

    # test perceptron model
    prediction = perceptron_predict(testset_values, w, eta)

    confusion_matrix = create_confusion_matrix(
        prediction, testset_labels)
    print('\nConfusion matrix: \n' + str(confusion_matrix) + '\n')

    # calculate accuracy
    accuracy = calculate_accuracy(prediction, testset_labels)
    print(f'Accuracy: {accuracy}\n')