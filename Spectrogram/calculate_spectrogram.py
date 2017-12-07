import numpy as np
from scipy import signal

def calculate_spectrogram_dataset(dataset):
    dataset_spectrogram = []
    for examples in dataset:
        canals = []
        for electrode_vector in examples:
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                calculate_spectrogram_vector(electrode_vector, npserseg=28, noverlap=20)
            #remove the low frequency signal as it's useless for sEMG (0-5Hz)
            spectrogram_of_vector = spectrogram_of_vector[1:]
            canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))

        example_to_classify = np.swapaxes(canals, 0, 1)
        dataset_spectrogram.append(example_to_classify)

    return dataset_spectrogram


def calculate_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window="hann",
                                                                                         scaling="spectrum")
    return spectrogram_of_vector, time_segment_sample, frequencies_samples

def show_spectrogram(frequencies_samples, time_segment_sample, spectrogram_of_vector):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 36})
    print np.shape(spectrogram_of_vector)
    print np.shape(time_segment_sample)
    print np.shape(frequencies_samples)

    time_segment_sample = [0., 65.,  130., 195., 250.]
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    print time_segment_sample
    plt.pcolormesh(time_segment_sample, frequencies_samples, spectrogram_of_vector)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [ms]')
    plt.title("STFT")
    plt.show()