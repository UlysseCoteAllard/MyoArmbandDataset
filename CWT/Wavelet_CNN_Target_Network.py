import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import SliceLayer
from PELU import pelu
from lasagne.layers.special import prelu
import time
import mc_dropout
from lasagne.layers import batch_norm
from scipy.stats import mode
import pickle
from lasagne.layers import ElemwiseSumLayer

class CNN_Progressif(object):
    def confusion_matrix(self, pred, Y):
        number_class = self._number_of_class
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

    def __getstate__(self):
        return lasagne.layers.get_all_param_values(self._network['output'])

    def __setstate__(self, weights):
        lasagne.layers.set_all_param_values(self._network['output'], weights)

    def get_state_pre_training(self):
        return lasagne.layers.get_all_param_values(self._network['output_gesture_pre_training'])

    def set_state_pre_training(self, weights):
        lasagne.layers.set_all_param_values(self._network['output_gesture_pre_training'], weights)

    def format_data(self, X_examples):
        return np.array(X_examples,dtype='float32')

    def cnn_separate_convolutions_pre_training(self, input, first_part, time=0):
        if first_part:
            self._network['input_first_part'] = input
            self._network['conv1_first_part_pre_training'] = pelu(batch_norm(lasagne.layers.Conv2DLayer(self._network['input_first_part'], num_filters=8,
                                                                           filter_size=(3, 3), stride=(1, 1),
                                                                           W=lasagne.init.HeNormal(gain='relu'))))
                                                            #prelu v2
            print self._network['conv1_first_part_pre_training'].output_shape


            if time == 0:
                name = 'first_part_dropout1_pre_training_0'
            elif time == 1:
                name = 'first_part_dropout1_pre_training_1'
            elif time == 2:
                name = 'first_part_dropout1_pre_training_2'
            else:
                name = 'first_part_dropout1_pre_training_3'
            self._network[name] = mc_dropout.MCDropout(
                self._network['conv1_first_part_pre_training'], p=self._percentage_dropout_training)

            self._network['conv2_first_part_pre_training'] = pelu(batch_norm(lasagne.layers.Conv2DLayer(
                self._network[name], num_filters=12,
                filter_size=(3, 3), W=lasagne.init.HeNormal(gain='relu'))))
            print self._network['conv2_first_part_pre_training'].output_shape

            first_dropout_2 = mc_dropout.MCDropout(self._network['conv2_first_part_pre_training'],
                                                   p=self._percentage_dropout_training)
            return first_dropout_2
        else:
            self._network['input_second_part'] = input
            self._network['conv1_second_part_pre_training'] = pelu(batch_norm(
                lasagne.layers.Conv2DLayer(self._network['input_second_part'], num_filters=24, filter_size=(3, 2),
                                           stride=(1, 1), W=lasagne.init.HeNormal(gain='relu'))))
            print self._network['conv1_second_part_pre_training'].output_shape

            second_dropout_1 = mc_dropout.MCDropout(self._network['conv1_second_part_pre_training'],
                                                    p=self._percentage_dropout_training)
            return second_dropout_1

    def pre_training_cnn(self):
        self._network = {}
        self._network['input_pre_training'] = lasagne.layers.InputLayer(shape=(None, self._number_of_channel,
                                                                                                8, 7),
                                                                                         input_var=self._x, pad='same',
                                                                                         W=lasagne.init.HeNormal(gain='relu'))
        self._network['input_normalized'] = prelu(batch_norm(self._network['input_pre_training']))
        print self._network['input_normalized'].output_shape

        first_part_input = SliceLayer(self._network['input_normalized'], indices=slice(0, 3), axis=1)
        print first_part_input.output_shape
        second_part_input = SliceLayer(self._network['input_normalized'], indices=slice(3, 6), axis=1)
        print second_part_input.output_shape

        third_part_input = SliceLayer(self._network['input_normalized'], indices=slice(6, 9), axis=1)
        print third_part_input.output_shape
        fourth_part_input = SliceLayer(self._network['input_normalized'], indices=slice(9, 12), axis=1)
        print fourth_part_input.output_shape

        first_dropout_2 = self.cnn_separate_convolutions_pre_training(first_part_input, first_part=True, time=0)
        second_dropout_2 = self.cnn_separate_convolutions_pre_training(second_part_input, first_part=True, time=1)

        third_dropout_2 = self.cnn_separate_convolutions_pre_training(third_part_input, first_part=True, time=2)
        fourth_dropout_2 = self.cnn_separate_convolutions_pre_training(fourth_part_input, first_part=True, time=3)

        self._network['concat_layer_1_pre_training'] = ElemwiseSumLayer([first_dropout_2, second_dropout_2])
        self._network['concat_layer_2_pre_training'] = ElemwiseSumLayer([third_dropout_2, fourth_dropout_2])

        self._network['last_fusion_layer1_pre_training'] = self.cnn_separate_convolutions(
            self._network['concat_layer_1_pre_training'], first_part=False)
        self._network['last_fusion_layer2_pre_training'] = self.cnn_separate_convolutions(
            self._network['concat_layer_2_pre_training'], first_part=False)

        self._network['concat_layer_final_pre_training'] = ElemwiseSumLayer(
            [self._network['last_fusion_layer1_pre_training'],
             self._network['last_fusion_layer2_pre_training']])

        self._network['conv3_pre_training_cnn'] = lasagne.layers.Conv2DLayer(
            self._network['concat_layer_final_pre_training'],
            num_filters=48,
            filter_size=(2, 2),
            W=lasagne.init.HeNormal(gain='relu'))

        self._network['conv3_pre_training'] = prelu(batch_norm(self._network['conv3_pre_training_cnn']))
        print self._network['conv3_pre_training'].output_shape

        self._network['dropout_3_pre_training'] = mc_dropout.MCDropout(self._network['conv3_pre_training'],
                                                                       p=self._percentage_dropout_training)

        self._network['pre_training_fc1_full'] = lasagne.layers.DenseLayer(self._network['dropout_3_pre_training'],
                                                                           num_units=100,
                                                                           W=lasagne.init.HeNormal(gain='relu'))

        self._network['fc1_pre_training'] = mc_dropout.MCDropout(
            prelu(batch_norm(self._network['pre_training_fc1_full'])),
            p=self._percentage_dropout_training)

        print self._network['fc1_pre_training'].output_shape

        self._network['pre_training_fc2_full'] = lasagne.layers.DenseLayer(self._network['fc1_pre_training'],
                                                                           num_units=100,
                                                                           W=lasagne.init.HeNormal(gain='relu'))

        self._network['fc2_pre_training'] = mc_dropout.MCDropout(
            prelu(batch_norm(self._network['pre_training_fc2_full'])), p=self._percentage_dropout_training)
        print self._network['fc2_pre_training'].output_shape

        self._network['output_gesture_pre_training'] = lasagne.layers.DenseLayer(self._network['fc2_pre_training'],
                                                                                 num_units=self._number_of_class,
                                                                                 nonlinearity=lasagne.nonlinearities.softmax,
                                                                                 W=lasagne.init.HeNormal(gain='relu'))
        print self._network['output_gesture_pre_training'].output_shape

        print "Pre-Training done printing"

    def cnn_separate_convolutions(self, input_signal, first_part, time=0):
        if first_part:
            conv1_first_part = pelu(batch_norm(lasagne.layers.Conv2DLayer(input_signal, num_filters=8,
                                                                           filter_size=(3, 3), stride=(1, 1),
                                                                           W=lasagne.init.HeNormal(gain='relu'))))
            print conv1_first_part.output_shape

            first_dropout_1 = mc_dropout.MCDropout(
                conv1_first_part, p=self._percentage_dropout_training)

            if time == 0:
                name = 'first_part_dropout1_pre_training_0'
            elif time == 1:
                name = 'first_part_dropout1_pre_training_1'
            elif time == 2:
                name = 'first_part_dropout1_pre_training_2'
            else:
                name = 'first_part_dropout1_pre_training_3'
            concat_first_part = ElemwiseSumLayer([lasagne.layers.ScaleLayer(self._network[name]), first_dropout_1])

            conv2_first_part = pelu(batch_norm(lasagne.layers.Conv2DLayer(
                concat_first_part, num_filters=12,
                filter_size=(3, 3), W=lasagne.init.HeNormal(gain='relu'))))
            print conv2_first_part.output_shape

            first_dropout_2 = mc_dropout.MCDropout(conv2_first_part, p=self._percentage_dropout_training)
            return first_dropout_2
        else:
            conv1_second_part = pelu(batch_norm(
                lasagne.layers.Conv2DLayer(input_signal, num_filters=24, filter_size=(3, 2),
                                           stride=(1, 1), W=lasagne.init.HeNormal(gain='relu'))))
            print conv1_second_part.output_shape

            second_dropout_1 = mc_dropout.MCDropout(conv1_second_part,
                                                    p=self._percentage_dropout_training)
            return second_dropout_1

    def cnn(self):
        self._network['input'] = pelu(batch_norm(lasagne.layers.InputLayer(shape=(None, self._number_of_channel,
                                                                                  8, 7),
                                                                           input_var=self._x, pad='same',
                                                                           W=lasagne.init.HeNormal(gain='relu'))))
        print self._network['input'].output_shape

        first_part_input = SliceLayer(self._network['input'], indices=slice(0, 3), axis=1)
        print first_part_input.output_shape
        second_part_input = SliceLayer(self._network['input'], indices=slice(3, 6), axis=1)
        print second_part_input.output_shape

        third_part_input = SliceLayer(self._network['input'], indices=slice(6, 9), axis=1)
        print third_part_input.output_shape
        fourth_part_input = SliceLayer(self._network['input'], indices=slice(9, 12), axis=1)
        print fourth_part_input.output_shape

        first_dropout_2 = self.cnn_separate_convolutions(first_part_input, first_part=True, time=0)
        second_dropout_2 = self.cnn_separate_convolutions(second_part_input, first_part=True, time=1)

        third_dropout_2 = self.cnn_separate_convolutions(third_part_input, first_part=True, time=2)
        fourth_dropout_2 = self.cnn_separate_convolutions(fourth_part_input, first_part=True, time=3)

        self._network['concat_layer_1'] = ElemwiseSumLayer([first_dropout_2, second_dropout_2,
                                                            lasagne.layers.ScaleLayer(self._network['concat_layer_1_pre_training'])])
        self._network['concat_layer_2'] = ElemwiseSumLayer([third_dropout_2, fourth_dropout_2,
                                                            lasagne.layers.ScaleLayer(self._network['concat_layer_2_pre_training'])])

        first_dropout_3 = self.cnn_separate_convolutions(self._network['concat_layer_1'], first_part=False)
        second_dropout_3 = self.cnn_separate_convolutions(self._network['concat_layer_2'], first_part=False)

        self._network['concat_layer'] = ElemwiseSumLayer([first_dropout_3, second_dropout_3,
                                                          lasagne.layers.ScaleLayer(self._network['concat_layer_final_pre_training'])])

        self._network['conv3'] = pelu(batch_norm(lasagne.layers.Conv2DLayer(self._network['concat_layer'],
                                                                            num_filters=48,
                                                                            filter_size=(2, 2),
                                                                            W=lasagne.init.HeNormal(gain='relu'))))

        print self._network['conv3'].output_shape

        self._network['dropout_3'] = mc_dropout.MCDropout(self._network['conv3'], p=self._percentage_dropout_training)

        self._network['merge_with_pre_training_dense_layer_1'] = ElemwiseSumLayer(
            [lasagne.layers.ScaleLayer(self._network['dropout_3_pre_training']), self._network['dropout_3']])
        print np.shape(self._network['pre_training_fc1_full'].W.get_value())
        self._network['fc1'] = mc_dropout.MCDropout(pelu(batch_norm(lasagne.layers.DenseLayer(
            self._network['merge_with_pre_training_dense_layer_1'], num_units=100, W=lasagne.init.HeNormal(gain='relu')))),
            p=self._percentage_dropout_training)

        print self._network['fc1'].output_shape

        self._network['merge_with_pre_training_dense_layer_2'] = ElemwiseSumLayer(
            [lasagne.layers.ScaleLayer(self._network['fc1_pre_training']), self._network['fc1']])

        self._network['fc2'] = mc_dropout.MCDropout(pelu(batch_norm(
            lasagne.layers.DenseLayer(self._network['merge_with_pre_training_dense_layer_2'], num_units=100,
                                      W=lasagne.init.HeNormal(gain='relu')))),
            p=self._percentage_dropout_training)

        print self._network['fc2'].output_shape

        self._network['merge_with_pre_training_output'] = ElemwiseSumLayer(
            [lasagne.layers.ScaleLayer(self._network['fc2_pre_training']), self._network['fc2']])

        self._network['output'] = lasagne.layers.DenseLayer(self._network['merge_with_pre_training_output'],
                                                            num_units=self._number_of_class,
                                                            nonlinearity=lasagne.nonlinearities.softmax,
                                                            W=lasagne.init.HeNormal(gain='relu'))

        print self._network['output'].output_shape

    def __init__(self, number_of_class, batch_size=300, number_of_channel=8, learning_rate=0.02,
                 dropout=.5):
        print('... Creating Classifier')
        self._learning_rate = learning_rate
        self._percentage_dropout_pre_training = dropout
        self._percentage_dropout_training = dropout

        self._percentage_dropout_dense_layers_pre_training = dropout
        self._percentage_dropout_cnn_layers_pre_training = dropout

        # create Theano variables for input and target minibatch
        self._x = T.tensor4('x')
        self._y_gesture = T.ivector('y_gesture')
        self._y_human = T.ivector('y_human')
        self._y_train = T.ivector('y_train')
        self._learning_rate_scalar = T.scalar(name='learning_rate')

        self._batchsize = batch_size
        self._number_of_class = number_of_class
        self._number_of_channel = number_of_channel
        self.pre_training_cnn()




        prediction_gesture_pre_training = lasagne.layers.get_output(self._network['output_gesture_pre_training'])


        self.loss_pre_training = lasagne.objectives.categorical_crossentropy(prediction_gesture_pre_training, self._y_gesture).mean()

        self.params_pre_training = lasagne.layers.get_all_params(self._network['output_gesture_pre_training'],
                                                                 trainable=True)
        self.updates_pre_training = lasagne.updates.adam(self.loss_pre_training, self.params_pre_training,
                                                         learning_rate=self._learning_rate_scalar)


        print self._network.values()

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        validation_prediction_pre_training = lasagne.layers.get_output(self._network['output_gesture_pre_training'], deterministic=True,
                                                          batch_norm_use_averages=True,
                                                          batch_norm_update_averages=False)

        test_loss = lasagne.objectives.categorical_crossentropy(validation_prediction_pre_training,
                                                                self._y_gesture)
        test_loss = test_loss.mean()

        # As a bonus, also create an expression for the classification accuracy:
        prediction_array = T.argmax(validation_prediction_pre_training, axis=1)

        test_acc = T.mean(T.eq(prediction_array, self._y_gesture),
                          dtype=theano.config.floatX)


        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_fn_pre_training = theano.function([self._x, self._y_gesture, self._learning_rate_scalar],
                                                     self.loss_pre_training, updates=self.updates_pre_training)

        # Compile a second function computing the validation loss and accuracy:
        self.val_fn_pre_training = theano.function([self._x, self._y_gesture], [test_loss, test_acc])

        # Fix the pre-trained network weights.
        for layer in lasagne.layers.get_all_layers(self._network['output_gesture_pre_training']):
            if layer.__class__.__name__ != 'BatchNormLayer':
                for param in layer.params:
                    layer.params[param].discard('trainable')
        self.cnn()
        weights_training = self.__getstate__()
        pickle.dump(weights_training, open("initialized_weights.p", "wb"))
        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        validation_prediction = lasagne.layers.get_output(self._network['output'], deterministic=True,
                                                          batch_norm_use_averages=True,
                                                          batch_norm_update_averages=False)


        prediction_gesture = lasagne.layers.get_output(self._network['output'])
        self.loss = lasagne.objectives.categorical_crossentropy(prediction_gesture, self._y_train).mean()

        self.params = lasagne.layers.get_all_params(self._network['output'],
                                                                 trainable=True)
        self.updates = lasagne.updates.adam(self.loss, self.params,
                                            learning_rate=self._learning_rate_scalar)

        test_loss = lasagne.objectives.categorical_crossentropy(validation_prediction,
                                                                self._y_train)
        test_loss = test_loss.mean()

        # As a bonus, also create an expression for the classification accuracy:
        prediction_array = T.argmax(validation_prediction, axis=1)

        test_acc = T.mean(T.eq(prediction_array, self._y_train),
                          dtype=theano.config.floatX)


        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_fn = theano.function([self._x, self._y_train, self._learning_rate_scalar], self.loss,
                                        updates=self.updates)

        # Compile a second function computing the validation loss and accuracy:
        self.val_fn = theano.function([self._x, self._y_train], [test_loss, test_acc])

        self._predict = lasagne.layers.get_output(self._network['output'], batch_norm_use_averages=True,
                                                  batch_norm_update_averages=False,
                                                  deterministic=True)
        self._prediction_array = theano.function([self._x], T.argmax(self._predict, axis=1))
        print "NUMBER OF PARAMETERS : "
        print lasagne.layers.count_params(self._network['output'])


    def predict(self, test_data):
        prediction = self._prediction_array(np.array(test_data, dtype=np.float32))
        return mode(prediction)[0]  # Return the class with the highest probability

    def iterate_minibatches(self, inputs, targets_1, batchsize, targets_2=None, shuffle=False,
                            do_multitache_learning=False):
        assert len(inputs) == len(targets_1)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if do_multitache_learning == False:
                yield inputs[excerpt], targets_1[excerpt]
            else:
                yield inputs[excerpt], targets_1[excerpt], targets_2[excerpt]

    def get_minibatch_input_only(self, inputs, batchsize, shuffle=False):
        indices = np.arange(len(inputs))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]

    def do_pre_training(self, datasets, n_epochs=500):
        initial_learning_rate = self._learning_rate
        train_set_x, train_set_y_gestures, train_set_y_humans = datasets[0]
        valid_set_x, valid_set_y_gestures, valid_set_y_humans = datasets[1]

        train_set_y_gestures_array = []
        valid_set_y_gestures_array = []
        train_set_y_humans_array = []
        valid_set_y_humans_array = []
        train_set_x_array = []
        valid_set_x_array = []
        for candidates_index in range(len(train_set_x)):
            train_set_x_array.append(np.array(train_set_x[candidates_index], dtype='float32'))
            train_set_y_gestures_array.append(np.array(train_set_y_gestures[candidates_index], dtype='int32'))
            train_set_y_humans_array.append(np.array(train_set_y_humans[candidates_index], dtype='int32'))

        for candidates_index in range(len(valid_set_x)):
            valid_set_x_array.append(np.array(valid_set_x[candidates_index], dtype='float32'))
            valid_set_y_gestures_array.append(np.array(valid_set_y_gestures[candidates_index], dtype='int32'))
            valid_set_y_humans_array.append(np.array(valid_set_y_humans[candidates_index], dtype='int32'))

        # Build an array of list of the batch norm layer parameters
        list_params_batch_norm_per_candidates = []
        for candidates_index in range(len(train_set_x)):
            candidate_batch_norm_param = []
            for layer in lasagne.layers.get_all_layers(self._network['output_gesture_pre_training']):
                if layer.__class__.__name__ == 'BatchNormLayer':
                    params_layer = layer.get_params()
                    list_value_params = []
                    for param in params_layer:
                        list_value_params.append(np.copy(param.get_value()))
                    candidate_batch_norm_param.append(list_value_params)
            list_params_batch_norm_per_candidates.append(candidate_batch_norm_param)


        # Finally, launch the training loop.
        print("Starting training...")
        best_validation_accuracy = -float('inf')
        patience = 10
        patience_increase = 30
        # We iterate over epochs:
        converged = False
        for epoch in range(n_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()

            # Generate blocks for training
            print np.shape(train_set_x_array)
            block_input = []
            block_target_gesture = []
            block_targets_human = []

            for i in range(len(train_set_x_array)):
                for batch in self.iterate_minibatches(train_set_x_array[i], train_set_y_gestures_array[i],
                                                      self._batchsize, shuffle=True,
                                                      targets_2=train_set_y_humans_array[i],
                                                      do_multitache_learning=True):
                    inputs, targets_gesture, targets_human = batch

                    block_input.append(inputs)
                    block_target_gesture.append(targets_gesture)
                    block_targets_human.append(targets_human)
            print np.shape(block_input)
            # Scramble de blocks
            random_vec = np.arange(len(block_input))
            np.random.shuffle(random_vec)

            for block_index in random_vec:
                current_human_index = block_targets_human[block_index][0]
                layers_param_values = list_params_batch_norm_per_candidates[current_human_index]
                # Set the current batch norm statistic to the one from the current candidate
                i = 0

                for layer in lasagne.layers.get_all_layers(self._network['output_gesture_pre_training']):
                    if layer.__class__.__name__ == 'BatchNormLayer':
                        layer_param_values = layers_param_values[i]
                        for p, v in zip(layer.get_params(), layer_param_values):
                            p.set_value(v)
                        i += 1

                inputs_block = block_input[block_index]
                targets_gesture_block = block_target_gesture[block_index]
                train_err += self.train_fn_pre_training(inputs_block, targets_gesture_block, self._learning_rate)
                train_batches += 1

                # Save the new found mean for this candidate
                candidate_batch_norm_param = []
                for layer in lasagne.layers.get_all_layers(self._network['output_gesture_pre_training']):
                    if layer.__class__.__name__ == 'BatchNormLayer':
                        params_layer = layer.get_params()
                        list_value_params = []
                        for param in params_layer:
                            list_value_params.append(np.copy(param.get_value()))
                        candidate_batch_norm_param.append(list_value_params)
                list_params_batch_norm_per_candidates[current_human_index] = candidate_batch_norm_param

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for i in range(len(valid_set_x_array)):
                layers_params = list_params_batch_norm_per_candidates[i]
                # Set the current batch norm statistic to the one from the current candidate
                j = 0
                for layer in lasagne.layers.get_all_layers(self._network['output_gesture_pre_training']):
                    if layer.__class__.__name__ == 'BatchNormLayer':
                        layer_param_values = layers_params[j]
                        for p, v in zip(layer.get_params(), layer_param_values):
                            p.set_value(v)
                        j += 1
                for batch in self.iterate_minibatches(valid_set_x_array[i], valid_set_y_gestures_array[i],
                                                      self._batchsize, targets_2=valid_set_y_humans_array[i],
                                                      shuffle=False,
                                                      do_multitache_learning=True):
                    inputs, targets_gesture, targets_human = batch
                    val_err_current = 0.
                    val_acc_current = 0.
                    number_try_val = 20
                    for i in range(number_try_val):
                        err, acc = self.val_fn_pre_training(inputs, targets_gesture)
                        val_err_current += err
                        val_acc_current += acc
                    val_acc_current /= number_try_val
                    val_err_current /= number_try_val
                    # err, acc = val_fn(inputs, targets)
                    val_err += val_err_current
                    val_acc += val_acc_current
                    print val_acc_current
                    val_batches += 1
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            if best_validation_accuracy < (val_acc / val_batches):
                best_validation_accuracy = (val_acc / val_batches)
                patience = patience_increase + epoch
                self.best_weights = self.get_state_pre_training()
                converged = False
                print "NEW BEST VALIDATION"
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

            if (patience <= epoch):
                print "Learning rate annealing!!!"
                self._learning_rate /= 5.
                print "New learning rate : ", self._learning_rate
                patience = patience_increase + epoch
                if converged:
                    break
                else:
                    converged = True
        print "BEST VALIDATION : ", best_validation_accuracy
        # Set the best weight in relation to the validation
        self.set_state_pre_training(self.best_weights)

        # Reset learning rate to its initial value in case of new training
        self._learning_rate = initial_learning_rate

    def fit(self, datasets, n_epochs=500):
        initial_learning_rate = self._learning_rate
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]

        train_set_y = np.array(train_set_y, dtype='int32')
        valid_set_y = np.array(valid_set_y, dtype='int32')

        train_set_x = np.array(train_set_x, dtype='float32')
        valid_set_x = np.array(valid_set_x, dtype='float32')

        # Finally, launch the training loop.
        print("Starting training...")
        best_validation_loss = float('inf')
        best_validation_accuracy = -float('inf')
        best_training_validation_loss = float('inf')
        patience = 30
        patience_increase = 5
        # We iterate over epochs:
        array_training_error = []
        array_validation_error = []
        converged = False
        for epoch in range(n_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(train_set_x, train_set_y, self._batchsize, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets, self._learning_rate)
                train_batches += 1
            array_training_error.append(train_err / train_batches)

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            number_try_val = 20
            for batch in self.iterate_minibatches(valid_set_x, valid_set_y, self._batchsize):
                inputs, targets = batch
                val_err_current = 0.
                val_acc_current = 0.
                for i in range(number_try_val):
                    err, acc = self.val_fn(inputs, targets)
                    val_err_current += err
                    val_acc_current += acc
                val_acc_current /= number_try_val
                val_err_current /= number_try_val
                val_err += val_err_current
                val_acc += val_acc_current
                val_batches += 1
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            array_validation_error.append(val_err/val_batches)
            if best_validation_loss > (val_err / val_batches):
                best_validation_loss = (val_err / val_batches)
                best_validation_accuracy = (val_acc / val_batches)
                patience = patience_increase + epoch
                self.best_weights = self.__getstate__()
                converged = False

                print "NEW BEST VALIDATION"
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

            if (patience <= epoch):
                print "Learning rate annealing!!!"
                self._learning_rate /= 5.
                print "New learning rate : ", self._learning_rate
                patience = patience_increase + epoch
                if converged:
                    break
                else:
                    converged = True
        print "BEST VALIDATION : ", best_validation_accuracy
        # Set the best weight in relation to the validation
        self.__setstate__(self.best_weights)
        # Reset learning rate to its initial value in case of new training
        self._learning_rate = initial_learning_rate
