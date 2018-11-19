import numpy as np
import load_evaluation_dataset
import source_network_raw_emg_enhanced
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
import copy


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


def calculate_fitness(examples_training, labels_training, examples_test0, labels_test0, examples_test1, labels_test_1,
                      learning_rate=.1, training_cycle=4):
    accuracy_test0 = []
    accuracy_test1 = []
    for j in range(17):
        print("CURRENT DATASET : ", j)
        examples_personne_training = []
        labels_gesture_personne_training = []
        
        for k in range(len(examples_training[j])):
            if k < training_cycle*7:
                examples_personne_training.extend(examples_training[j][k])
                labels_gesture_personne_training.extend(labels_training[j][k])

        X_test_0, Y_test_0 = [], []
        for k in range(len(examples_test0)):
            X_test_0.extend(examples_test0[j][k])
            Y_test_0.extend(labels_test0[j][k])

        X_test_1, Y_test_1 = [], []
        for k in range(len(examples_test1)):
            X_test_1.extend(examples_test1[j][k])
            Y_test_1.extend(labels_test_1[j][k])
        
        print(np.shape(examples_personne_training))
        examples_personne_scrambled, labels_gesture_personne_scrambled = scramble(examples_personne_training,
                                                                                  labels_gesture_personne_training)
        valid_examples = examples_personne_scrambled[0:int(len(examples_personne_scrambled) * 0.1)]
        labels_valid = labels_gesture_personne_scrambled[0:int(len(labels_gesture_personne_scrambled) * 0.1)]
        
        X_fine_tune = examples_personne_scrambled[int(len(examples_personne_scrambled) * 0.1):]
        Y_fine_tune = labels_gesture_personne_scrambled[int(len(labels_gesture_personne_scrambled) * 0.1):]
        
        train = TensorDataset(torch.from_numpy(np.array(X_fine_tune, dtype=np.float32)),
                              torch.from_numpy(np.array(Y_fine_tune, dtype=np.int64)))
        
        validation = TensorDataset(torch.from_numpy(np.array(valid_examples, dtype=np.float32)),
                                   torch.from_numpy(np.array(labels_valid, dtype=np.int64)))
        
        trainloader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
        validationloader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=True, drop_last=True)
        
        cnn = source_network_raw_emg_enhanced.Net(number_of_class=7).cuda()
        
        criterion = nn.CrossEntropyLoss(size_average=False)
        optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
        
        precision = 1e-6
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                         verbose=True, eps=precision)
        
        cnn = train_model(cnn, criterion, optimizer, scheduler, dataloaders={"train": trainloader,
                                                                             "val": validationloader},
                          precision=precision)
        
        cnn.eval()
        X_test_0, Y_test_0 = scramble(X_test_0, Y_test_0)
        
        test_0 = TensorDataset(torch.from_numpy(np.array(X_test_0, dtype=np.float32)),
                               torch.from_numpy(np.array(Y_test_0, dtype=np.int64)))

        X_test_1, Y_test_1 = scramble(X_test_1, Y_test_1)

        test_1 = TensorDataset(torch.from_numpy(np.array(X_test_1, dtype=np.float32)),
                               torch.from_numpy(np.array(Y_test_1, dtype=np.int64)))
        
        test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=256, shuffle=False)
        total = 0
        correct_prediction_test_0 = 0
        for k, data_test_0 in enumerate(test_0_loader, 0):
            # get the inputs
            inputs_test_0, ground_truth_test_0 = data_test_0
            inputs_test_0, ground_truth_test_0 = Variable(inputs_test_0.cuda()), Variable(ground_truth_test_0.cuda())
            
            outputs_test_0 = cnn(inputs_test_0)
            _, predicted = torch.max(outputs_test_0.data, 1)
            correct_prediction_test_0 += (predicted.cpu().numpy() == ground_truth_test_0.data.cpu().numpy()).sum()
            total += ground_truth_test_0.size(0)
        print("ACCURACY TEST_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))
        accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))

        test_1_loader = torch.utils.data.DataLoader(test_1, batch_size=256, shuffle=False)
        total = 0
        correct_prediction_test_1 = 0
        for k, data_test_1 in enumerate(test_1_loader, 0):
            # get the inputs
            inputs_test_1, ground_truth_test_1 = data_test_1
            inputs_test_1, ground_truth_test_1 = Variable(inputs_test_1.cuda()), Variable(ground_truth_test_1.cuda())
    
            outputs_test_1 = cnn(inputs_test_1)
            _, predicted = torch.max(outputs_test_1.data, 1)
            correct_prediction_test_1 += (predicted.cpu().numpy() == ground_truth_test_1.data.cpu().numpy()).sum()
            total += ground_truth_test_1.size(0)
        print("ACCURACY TEST_1 FINAL : %.3f %%" % (100 * float(correct_prediction_test_1) / float(total)))
        accuracy_test1.append(100 * float(correct_prediction_test_1) / float(total))
        
    print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_test0).mean())
    print("AVERAGE ACCURACY TEST 1 %.3f" % np.array(accuracy_test1).mean())
    return accuracy_test0, accuracy_test1


def train_model(cnn, criterion, optimizer, scheduler, dataloaders, num_epochs=500, precision=1e-8):
    since = time.time()
    
    best_loss = float('inf')
    
    patience = 30
    patience_increase = 10
    
    best_weights = copy.deepcopy(cnn.state_dict())
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn.train(True)  # Set model to training mode
            else:
                cnn.train(False)  # Set model to evaluate mode
            
            running_loss = 0.
            running_corrects = 0
            total = 0
            
            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                
                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    cnn.train()
                    # forward
                    outputs = cnn(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    # backward
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                
                else:
                    cnn.eval()
                    # forward
                    outputs = cnn(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss = loss.item()
            
                # statistics
                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)
                    
            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total
            
            print('{} Loss: {} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(cnn.state_dict())
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
        if epoch >= 25:
            break
    print()
    
    time_elapsed = time.time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    cnn.load_state_dict(copy.deepcopy(best_weights))
    cnn.eval()
    return cnn


if __name__ == '__main__':
    # Change the path of the Evaluation Dataset to where you have it downloaded
    
    # Comment between here
    
    examples, labels = load_evaluation_dataset.read_data('EvaluationDataset', type="training0")
    datasets = [examples, labels]

    np.save("formatted_datasets/saved_evaluation_dataset_training.npy", datasets)
    
    examples, labels = load_evaluation_dataset.read_data('EvaluationDataset', type="Test0")
    datasets = [examples, labels]

    np.save("formatted_datasets/saved_evaluation_dataset_test0.npy", datasets)
    
    examples, labels = load_evaluation_dataset.read_data('EvaluationDataset', type="Test1")
    datasets = [examples, labels]

    np.save("formatted_datasets/saved_evaluation_dataset_test1.npy", datasets)
    
    # And here if the pre-training dataset was already processed and saved
    
    # Comment between here
    
    datasets_training = np.load("formatted_datasets/saved_evaluation_dataset_training.npy", encoding="bytes")
    examples_training, labels_training = datasets_training
    
    datasets_test0 = np.load("formatted_datasets/saved_evaluation_dataset_test0.npy", encoding="bytes")
    examples_test0, labels_test0 = datasets_test0
    
    datasets_test1 = np.load("formatted_datasets/saved_evaluation_dataset_test1.npy", encoding="bytes")
    examples_test1, labels_test1 = datasets_test1
    
    # And here if the pre-training of the network was already completed.
    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []
    # learning_rate = 0.002335721469090121
    for training_cycle in range(1, 5):
        test_0 = []
        test_1 = []
        for i in range(20):
            accuracy_test0, accuracy_test1 = calculate_fitness(examples_training, labels_training, examples_test0,
                                                               labels_test0, examples_test1, labels_test1,
                                                               learning_rate=0.002335721469090121,
                                                               training_cycle=training_cycle)
    
            test_0.append(accuracy_test0)
            test_1.append(accuracy_test1)
            print("TEST 0 SO FAR: ", test_0)
            print("TEST 1 SO FAR: ", accuracy_test1)
            print("CURRENT AVERAGE : ", (np.mean(test_0)+np.mean(test_1))/2.)
        
        print("ACCURACY FINAL TEST 0: ", test_0)
        print("ACCURACY FINAL TEST 0: ", np.mean(test_0))
        print("ACCURACY FINAL TEST 1: ", test_1)
        print("ACCURACY FINAL TEST 1: ", np.mean(test_1))
        print("ACCURACY FINAL: ", (np.mean(test_0)+np.mean(test_1))/2.)
        
        with open("results/evaluation_dataset_convnet_enhanced.txt", "a") as myfile:
            myfile.write("ConvNet Training Cycle : "+str(training_cycle)+"\n\n")
            myfile.write("Test 0: \n")
            myfile.write(str(test_0) + '\n')
            myfile.write(str(np.mean(test_0, axis=0)) + '\n')
            myfile.write(str(np.mean(test_0)) + '\n')
            myfile.write("Test 1: \n")
            myfile.write(str(test_1) + '\n')
            myfile.write(str(np.mean(test_1, axis=0)) + '\n')
            myfile.write(str(np.mean(test_1)) + '\n')
            myfile.write("Test Mean: \n")
            myfile.write(str(np.mean(test_0, axis=0)) + '\n')
            myfile.write(str((np.mean(test_0) + np.mean(test_1)) / 2.) + '\n')
            myfile.write("\n\n\n")
