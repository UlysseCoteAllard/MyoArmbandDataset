import numpy as np
import pywt
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def calculate_wavelet_dataset(dataset):
    dataset_spectrogram = []
    mother_wavelet = 'mexh'
    for examples in dataset:
        canals = []
        for electrode_vector in examples:
            coefs = calculate_wavelet_vector(np.abs(electrode_vector), mother_wavelet=mother_wavelet, scales=np.arange(1, 33))  # 33 originally
            print np.shape(coefs)
            show_wavelet(coef=coefs)
            coefs = zoom(coefs, .25, order=0)
            coefs = np.delete(coefs, axis=0, obj=len(coefs)-1)
            coefs = np.delete(coefs, axis=1, obj=np.shape(coefs)[1]-1)
            canals.append(np.swapaxes(coefs, 0, 1))
        example_to_classify = np.swapaxes(canals, 0, 1)
        dataset_spectrogram.append(example_to_classify)

    return dataset_spectrogram

def calculate_wavelet_vector(vector, mother_wavelet='mexh', scales=np.arange(1, 32)):
    coef, freqs = pywt.cwt(vector, scales=scales, wavelet=mother_wavelet)
    return coef

def show_wavelet(coef):
    print np.shape(coef)
    plt.rcParams.update({'font.size': 36})
    plt.matshow(coef)
    plt.ylabel('Scale')
    plt.xlabel('Samples')
    plt.show()

