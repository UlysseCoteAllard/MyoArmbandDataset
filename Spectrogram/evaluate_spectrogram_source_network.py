import load_evaluation_dataset
import numpy as np
import Spectrogram_CNN_Source_Network
import pickle
from sklearn.metrics import accuracy_score

def confusion_matrix(pred, Y, number_class=7):
    confusion_matrice = []
    for x in range(0, number_class):
        vector = []
        for y in range(0, number_class):
            vector.append(0)
        confusion_matrice.append(vector)
    for prediction, real_value in zip(pred, Y):
        prediction = int(prediction)
        real_value = int(real_value)
        confusion_matrice[prediction][real_value] += 1
    return np.array(confusion_matrice)

def scramble(examples, labels, second_labels=[]):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    if len(second_labels) == len(labels):
        new_second_labels = []
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
            new_second_labels.append(second_labels[i])
        return new_examples, new_labels, new_second_labels
    else:
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
        return new_examples, new_labels

def calculate_fitness(examples_training, labels_training, examples_validation0, labels_validation0,
                      examples_validation1, labels_validation1, cnn):
    accuracy_test0 = []
    accuracy_test1 = []
    initialized_weights = np.load("initialized_weights.npy")
    for i in range(17):
        X_fine_tune_train, Y_fine_tune_train = [], []
        for j in range(len(labels_training)):
            if j == i:
                print "Current dataset test : ", i
                for k in range(len(examples_training[j])):
                    if (k < 28):
                        X_fine_tune_train.extend(examples_training[j][k])
                        Y_fine_tune_train.extend(labels_training[j][k])
        X_test_0, Y_test_0 = [], []
        for j in range(len(labels_validation0)):
            if j == i:
                for k in range(len(examples_validation0[j])):
                    X_test_0.extend(examples_validation0[j][k])
                    Y_test_0.extend(labels_validation0[j][k])

        X_test_1, Y_test_1 = [], []
        for j in range(len(labels_validation1)):
            if j == i:
                for k in range(len(examples_validation1[j])):
                    X_test_1.extend(examples_validation1[j][k])
                    Y_test_1.extend(labels_validation1[j][k])

        # Do a first training to learn the weights of the whole network.

        X_fine_tune, Y_fine_tune = scramble(X_fine_tune_train, Y_fine_tune_train)
        valid_examples = X_fine_tune[0:int(len(X_fine_tune) * 0.1)]
        labels_valid = Y_fine_tune[0:int(len(Y_fine_tune) * 0.1)]

        X_fine_tune = X_fine_tune[int(len(X_fine_tune) * 0.1):]
        Y_fine_tune = Y_fine_tune[int(len(Y_fine_tune) * 0.1):]

        print "SHAPE: ", np.shape(X_fine_tune)

        datasets = [(X_fine_tune, Y_fine_tune), (valid_examples, labels_valid)]
        cnn.__setstate__(initialized_weights)
        cnn.fit(datasets)

        predictions = []
        for example in X_test_0:
            example_repeated = []
            for i in range(20):
                example_repeated.append(example)
            predictions.append(cnn.predict(example_repeated))
        score2 = accuracy_score(Y_test_0, predictions)
        print score2
        accuracy_test0.append(score2)
        print "AVERAGE SO FAR (TEST0) : ", np.mean(accuracy_test0)

        predictions = []
        for example in X_test_1:
            example_repeated = []
            for i in range(20):
                example_repeated.append(example)
            predictions.append(cnn.predict(example_repeated))
        score2 = accuracy_score(Y_test_1, predictions)
        print score2
        accuracy_test1.append(score2)
        print "AVERAGE SO FAR (TEST1) : ", np.mean(accuracy_test1)

        print "ACCURACY SO FAR TOTAL : ", (np.mean(accuracy_test0) + np.mean(accuracy_test1)) / 2.
    return accuracy_test0, accuracy_test1


# Comment between here
'''
examples, labels = load_evaluation_dataset.read_data('../EvaluationDataset',
                                            type='training0')

datasets = [examples, labels]
pickle.dump(datasets, open("saved_dataset_training.p", "wb"))

examples, labels = load_evaluation_dataset.read_data('../EvaluationDataset',
                                            type='Test0')

datasets = [examples, labels]
pickle.dump(datasets, open("saved_dataset_test0.p", "wb"))

examples, labels = load_evaluation_dataset.read_data('../EvaluationDataset',
                                            type='Test1')

datasets = [examples, labels]
pickle.dump(datasets, open("saved_dataset_test1.p", "wb"))
'''
# and here if the evaluation dataset was already processed and saved with "load_evaluation_dataset"


datasets_training = np.load("saved_dataset_training.p")
examples_training, labels_training = datasets_training

datasets_validation0 = np.load("saved_dataset_test0.p")
examples_validation0, labels_validation0 = datasets_validation0

datasets_validation1 = np.load("saved_dataset_test1.p")
examples_validation1, labels_validation1 = datasets_validation1
print "SHAPE", np.shape(examples_training)

accuracy_one_by_one = []
array_training_error = []
array_validation_error = []


test_0 = []
test_1 = []

for i in range(20):
    cnn = Spectrogram_CNN_Source_Network.CNN_Progressif(number_of_class=7, batch_size=128, number_of_channel=4,
                                                        learning_rate=0.00681292, dropout=.5)
    accuracy_subject, accuracy_one_one = calculate_fitness(examples_training, labels_training,
                                                           examples_validation0, labels_validation0,
                                                           examples_validation1, labels_validation1, cnn=cnn)
    print accuracy_subject
    test_0.append(accuracy_subject)
    test_1.append(accuracy_one_one)
    print "TEST 0 SO FAR: ", test_0
    print "TEST 1 SO FAR: ", test_1
    print "CURRENT AVERAGE : ", (np.mean(test_0) + np.mean(test_1))/2.

print "ACCURACY FINAL TEST 0: ", test_0
print "ACCURACY FINAL TEST 0: ", np.mean(test_0)
print "ACCURACY FINAL TEST 1: ", test_1
print "ACCURACY FINAL TEST 1: ", np.mean(test_1)

with open("results_spectrogram_4_cycles.txt", "a") as myfile:
    myfile.write("CNN STFT: \n\n")
    myfile.write("Test 0: \n")
    myfile.write(str(np.mean(test_0, axis=0)) +'\n')
    myfile.write(str(np.mean(test_0)) +'\n')

    myfile.write("Test 1: \n")
    myfile.write(str(np.mean(test_1, axis=0)) +'\n')
    myfile.write(str(np.mean(test_1)) +'\n')
    myfile.write("\n\n\n")