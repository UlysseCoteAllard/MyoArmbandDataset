import load_pre_training_dataset
import load_evaluation_dataset
import numpy as np
import Wavelet_CNN_Target_Network
import cPickle as pickle
from sklearn.metrics import accuracy_score
import msgpack_numpy as m
m.patch()

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

def calculate_pre_training(examples, labels, cnn):
    X_training, Y_gesture, Y_human = [], [], []
    X_validation, Y_gesture_validation, Y_human_validation = [], [], []
    human_number = 0
    for j in range(19):
        examples_personne_training = []
        labels_gesture_personne_training = []
        labels_human_personne_training = []

        examples_personne_valid = []
        labels_gesture_personne_valid = []
        labels_human_personne_valid = []

        for k in range(len(examples[j])):
            if k < 21:
                examples_personne_training.extend(examples[j][k])
                labels_gesture_personne_training.extend(labels[j][k])
                labels_human_personne_training.extend(human_number * np.ones(len(labels[j][k])))
            else:
                examples_personne_valid.extend(examples[j][k])
                labels_gesture_personne_valid.extend(labels[j][k])
                labels_human_personne_valid.extend(human_number * np.ones(len(labels[j][k])))

        print np.shape(examples_personne_training)
        examples_personne_scrambled, labels_gesture_personne_scrambled, labels_human_personne_scrambled = scramble(
            examples_personne_training, labels_gesture_personne_training, labels_human_personne_training)

        examples_personne_scrambled_valid, labels_gesture_personne_scrambled_valid, labels_human_personne_scrambled_valid = scramble(
            examples_personne_valid, labels_gesture_personne_valid, labels_human_personne_valid)

        X_training.append(examples_personne_scrambled)
        Y_gesture.append(labels_gesture_personne_scrambled)
        Y_human.append(labels_human_personne_scrambled)

        X_validation.append(examples_personne_scrambled_valid)
        Y_gesture_validation.append(labels_gesture_personne_scrambled_valid)
        Y_human_validation.append(labels_human_personne_scrambled_valid)
        human_number += 1
    print "Shape training : ", np.shape(X_training[0])
    print "Shape valid : ", np.shape(X_validation[0])

    datasets = [(X_training, Y_gesture, Y_human),
                (X_validation, Y_gesture_validation, Y_human_validation)]

    cnn.do_pre_training(datasets)
    weights_pre_training = cnn.get_state_pre_training()
    pickle.dump(weights_pre_training, open("target_network_pre_trained_weights.p", "wb"))

    weight_pre_training = pickle.load(open("target_network_pre_trained_weights.p"))
    cnn.set_state_pre_training(weight_pre_training)
    print "PRE TRAINING DONE"

def calculate_fitness(examples_training, labels_training, examples_validation0, labels_validation0,
                      examples_validation1, labels_validation1):
    cnn = Wavelet_CNN_Target_Network.CNN_Progressif(number_of_class=7, batch_size=128,
                                                                   number_of_channel=12,
                                                                   learning_rate=0.0404709, dropout=.5)
    weight_pre_training = pickle.load(open("target_network_pre_trained_weights.p"))
    initialized_weights = pickle.load(open("initialized_weights.p"))

    accuracy_test0 = []
    accuracy_test1 = []
    #for i in [5, 11, 13, 15, 16]:
    for i in range(18):
        X_fine_tune_train, Y_fine_tune_train = [], []
        for j in range(len(labels_training)):
            if j == i:
                print "Current dataset test : ", i
                for k in range(len(examples_training[j])):
                    # Change this condition (7, 14, 21 or 28) to have 1, 2 ,3 and 4 cycle of data to train the network
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
        print np.shape(X_fine_tune)

        datasets = [(X_fine_tune, Y_fine_tune), (valid_examples, labels_valid)]
        cnn.__setstate__(initialized_weights)
        cnn.set_state_pre_training(weight_pre_training)
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



examples, labels = load_evaluation_dataset.read_data('../EvaluationDataset',
                                            type='training0')

datasets = [examples, labels]
np.save("saved_dataset_training.npy", datasets)

examples, labels = load_evaluation_dataset.read_data('../EvaluationDataset',
                                            type='Validation0')

datasets = [examples, labels]
np.save("saved_dataset_test0.npy", datasets)

examples, labels = load_evaluation_dataset.read_data('../EvaluationDataset',
                                            type='Validation1')

datasets = [examples, labels]
np.save("saved_dataset_test1.npy", datasets)


# Comment between here

examples, labels = load_pre_training_dataset.read_data('../PreTrainingDataset')
datasets = [examples, labels]

np.save("saved_pre_training_dataset.npy", datasets)

# And here if the pre-training dataset was already processed and saved

# Comment between here

datasets_pre_training = np.load("saved_pre_training_dataset.npy")
examples_pre_training, labels_pre_training = datasets_pre_training

cnn = Wavelet_CNN_Target_Network.CNN_Progressif(number_of_class=7, batch_size=1312, number_of_channel=12,
                                                       learning_rate=0.0404709, dropout=.35)

calculate_pre_training(examples_pre_training, labels_pre_training, cnn=cnn)

# And here if the pre-training of the network was already completed.

datasets_training = np.load("saved_dataset_training.npy")

examples_training, labels_training = datasets_training

datasets_validation0 = np.load("saved_dataset_test0.npy")

examples_validation0, labels_validation0 = datasets_validation0

datasets_validation1 = np.load("saved_dataset_test1.npy")


examples_validation1, labels_validation1 = datasets_validation1

print "SHA", np.shape(examples_training)

accuracy_one_by_one = []
array_training_error = []
array_validation_error = []

test_0 = []
test_1 = []
for i in range(20):
    accuracy_subject, accuracy_one_one = calculate_fitness(examples_training, labels_training,
                                                           examples_validation0, labels_validation0,
                                                           examples_validation1, labels_validation1)
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

with open("results_CWT_transfer_learning_1_cycles.txt", "a") as myfile:
    myfile.write("CNN STFT: \n\n")
    myfile.write("Test 0: \n")
    myfile.write(str(np.mean(test_0, axis=0)) +'\n')
    myfile.write(str(np.mean(test_0, axis=1)) +'\n')
    myfile.write(str(np.mean(test_0)) +'\n')

    myfile.write("Test 1: \n")
    myfile.write(str(np.mean(test_1, axis=0)) +'\n')
    myfile.write(str(np.mean(test_1, axis=1)) + '\n')
    myfile.write(str(np.mean(test_1)) +'\n')
    myfile.write("\n\n\n")