# MyoArmbandDataset
The paper describing this dataset is available here: https://arxiv.org/abs/1801.07756

Required libraries: 

Theano http://deeplearning.net/software/theano/ 

Lasagne https://lasagne.readthedocs.io/en/latest/

Scikit-learn http://scikit-learn.org/stable/

SciPy https://www.scipy.org/

PyWavelets https://pywavelets.readthedocs.io/en/latest/

Matplotlib https://matplotlib.org/


Installation through Anaconda is the easiest and fastest way to have the code running (https://conda.io/docs/user-guide/install/download.html). 

The files that should be utilized to launch the experiments are evaluate_spectrogram_source_network.py (no transfer learning) and evaluate_spectrogram_target_network.py (with transfer learning). Similar files are available for the CWT-based ConvNet. 


Dataset:
The dataset is separated in two subdatasets (pre-training and evaluation dataset). The datasets contain a folder per subject. The folder training0 (for the pre-training dataset) and the folders training0, test0 and test1 (for the evaluation dataset) contain the raw myo armband signal in files named classe_i.dat where i goes from 0 to 27. Each file contain a the sEMG signal for a specific gestures. In order: 0 = Neutral, 1 = Radial Deviation, 2 = Wrist Flexion, 3 = Ulnar Deviation, 4 = Wrist Extension, 5 = Hand Close, 6 = Hand Open. The gestures than cycles in the same order (i.e. 7 = Neutral, 8 = Radial Deviation, etc). 

Examples to load the datasets are given with the files (load_pre_training_dataset.py and load_evaluation_dataset.py). Note that for pre-training the datasets employed in the paper are: the first 7 women recording (0 through 6 inclusively) and the first 12 men recording (0 through 11 inclusively) from the pre-training dataset. For the evaluation datasets, the datasets employed to generate the results are: the first two women (0 through 1 inclusively) and the first 15 men (0 through 14 inclusively). The additionals datasets were added at a later date (total of 22 participants for the pre-training dataset and 18 for the evaluation dataset). 

The data acquisition protocol was approved by the Laval University Ethics committee (approbation number: 2017-026/21-02-2016).

This work is based on: 
Cote-Allard, Ulysse, Cheikh Latyr Fall, Alexandre Campeau-Lecours, Clément Gosselin, François Laviolette, and Benoit Gosselin. "Transfer learning for sEMG hand gestures recognition using convolutional neural networks." In 2017 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pp. 1663-1668. IEEE, 2017.

and 

Côté-Allard, Ulysse, Cheikh Latyr Fall, Alexandre Drouin, Alexandre Campeau-Lecours, Clément Gosselin, Kyrre Glette, François Laviolette, and Benoit Gosselin. "Deep learning for electromyographic hand gesture signal classification using transfer learning." IEEE Transactions on Neural Systems and Rehabilitation Engineering 27, no. 4 (2019): 760-771.

Note that part of this work has been ported from Theano to PyTorch (From Python 2.7 to Python 3.6). The structure of the project within PyTorchImplementation is the same as previously described. Note also that the currently available PyTorch networks do not employ the PELU activation function, as my current implementation is too slow. As a result the Target Network perform slightly worst then the one reported in the article (around 98.23% accuracy compare to the 98.31% accuracy with PELU). Apart from that, the implementation should be identical.  
