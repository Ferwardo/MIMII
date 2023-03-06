import matplotlib.pyplot as plt
import numpy as np
import librosa.feature
import librosa.display
from scipy import signal
from scipy.signal import get_window
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from FE_funcs.feat_extract import FE
import os

__all__ = ["frequencyPlot", "spectogram", "MFCC"]


def frequencyPlot(filename, filepath=""):
    # Load in the WAV file
    samplerate, data = wavfile.read(filepath + "/" + filename)

    # Do the FFT
    values = fft(data)
    frequencies = fftfreq(data.shape[0], 1 / samplerate)

    # Show the frequency plot for each channel
    for i in range(0, values.shape[1]):
        plt.title(f"Channel: {i}")
        plt.plot(frequencies, np.abs(values[:, i]))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Value")
        plt.show()


def spectogram(filename, filepath=""):
    # Load in the WAV file
    samplerate, data = wavfile.read(filepath + "/" + filename)

    # Generate a spectogram for each channel of the audio file
    for i in range(0, data.shape[1]):
        frequencies, segmentTimes, Sxx = signal.spectrogram(data[:, i], samplerate)

        plt.title(f"Channel: {i}")
        plt.pcolormesh(segmentTimes, frequencies, Sxx, shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.colorbar()
        plt.show()


def MFCC(filename, filepath="", mode="feat_extract", fft_size=2048, hop_size=15, window_shape="hann", saveLogMel=False,
         saveMFCC=False):
    if mode == "librosa":
        MFCCLibrosa(filename, filepath=filepath)
    elif mode == "scipy":
        MFCCScipy(filename, filepath=filepath, fft_size=fft_size, hop_size=hop_size, window_shape=window_shape)
    elif mode == "feat_extract":
        return MFCCFeatExtract(filename, filepath=filepath, saveLogMel=saveLogMel, saveMFCC=saveMFCC)


def MFCCFeatExtract(filename, filepath="", saveLogMel=False, saveMFCC=False):
    # Load in the WAV file
    samplerate, data = wavfile.read(filepath + "/" + filename)

    # Define feature extraction parameters
    featconf = {}
    featconf['dcRemoval'] = 'hpf'
    featconf['samFreq'] = samplerate
    featconf['lowFreq'] = 0
    featconf['highFreq'] = featconf['samFreq'] / 2
    featconf['stepSize_ms'] = 10
    featconf['frameSize_ms'] = 32
    featconf['melSize'] = 64

    featextract = FE(featconf)

    # A numpy array for the logmel features and mfccs from all channels
    logmelAllChannels = np.empty(
        shape=(data.shape[1],
               int(np.floor(((np.shape(data)[0] - (featconf['frameSize'] - 1) - 1) / featconf['stepSize']) + 1)),
               featconf["melSize"]))
    mfccAllChannels = np.empty(shape=(data.shape[1], 997, 20))

    for i in range(0, data.shape[1]):
        logmelframes = featextract.fe_transform(data[:, i])
        logmelAllChannels[i] = logmelframes

        dct_filters = dct(20, featconf["melSize"])

        mfcc = np.dot(logmelframes, dct_filters.transpose())
        mfccAllChannels[i] = mfcc

    if saveLogMel:
        os.makedirs(filepath + "/logmel/", exist_ok=True)
        np.save(filepath + "/logmel/" + filename.replace(".wav", ".npy"), logmelAllChannels)
    if saveMFCC:
        os.makedirs(filepath + "/mfcc/", exist_ok=True)
        np.save(filepath + "/mfcc/" + filename.replace(".wav", ".npy"), mfccAllChannels)

    return mfccAllChannels


def MFCCScipy(filename, filepath="", fft_size=2048, hop_size=15, window_shape="hann"):
    # originally used https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial/notebook

    # Load in the WAV file
    samplerate, data = wavfile.read(filepath + "/" + filename)

    # get the window for later
    window = get_window(window_shape, fft_size, fftbins=True)
    plt.figure(figsize=(15, 4))
    plt.plot(window)
    plt.grid(True)
    plt.show()

    # Do it for each channel
    for i in range(0, data.shape[1]):
        framedAudio = frame_audio(data[:, i], fft_size, hop_size, samplerate)

        print(f"Channel {i} framed audio shape: {framedAudio.shape}")

        # window the audio
        windowedAudio = framedAudio * window

        # FFT for channel
        freqSpectrum = np.empty((int(1 + fft_size // 2), np.transpose(windowedAudio).shape[1]), dtype=np.complex64,
                                order='F')

        for n in range(freqSpectrum.shape[1]):
            freqSpectrum[:, n] = fft(np.transpose(windowedAudio)[:, n], axis=0)[:freqSpectrum.shape[0]]

        freqSpectrum = np.transpose(freqSpectrum)

        # Get the power of the signal
        powerSpectrum = np.square(np.abs(freqSpectrum))

        # MEL-spaced filterbank
        minFrequency = 0
        maxFrequency = samplerate / 2
        melFilterNumber = 10

        filter_points, mel_freqs = get_filter_points(minFrequency, maxFrequency, melFilterNumber, fft_size,
                                                     sample_rate=samplerate)

        # get the filters
        filters = get_filters(filter_points, fft_size)

        enorm = 2.0 / (mel_freqs[2:melFilterNumber + 2] - mel_freqs[:melFilterNumber])
        filters *= enorm[:, np.newaxis]

        # Filter the audio
        filteredAudio = np.dot(filters, np.transpose(powerSpectrum))
        logAudio = 10.0 * np.log10(filteredAudio)

        # generate Cepstral Coefficients
        dct_filter_num = 40

        dct_filters = dct(dct_filter_num, melFilterNumber)

        cepstral_coefficents = np.dot(dct_filters, logAudio)

        plt.figure(figsize=(15, 5))
        plt.title(f"Channel: {i}")
        plt.plot(np.linspace(0, len(data) / samplerate, num=len(data)), data)
        plt.imshow(cepstral_coefficents, aspect='auto', origin='lower')
        plt.show()


def MFCCLibrosa(filename, filepath=""):
    # Load in the WAV file
    samplerate, data = wavfile.read(filepath + "/" + filename)

    for i in range(0, data.shape[1]):
        mfcc = librosa.feature.mfcc(y=data[:, i].astype('float64'), sr=samplerate, n_mfcc=20, hop_length=1024,
                                    htk=True)

        S = librosa.feature.melspectrogram(y=data[:, i].astype('float64'), sr=samplerate, n_mels=128,
                                           fmax=8000)

        # visualise
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                       x_axis='time', y_axis='mel', fmax=8000,
                                       ax=ax[0])
        fig.colorbar(img, ax=[ax[0]])
        ax[0].set(title='Mel spectrogram')
        ax[0].label_outer()
        img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
        fig.colorbar(img, ax=[ax[1]])
        ax[1].set(title='MFCC')
        plt.show()


# private functions
def frame_audio(audio, fft_size=2048, hop_size=10, sample_rate=44100):
    # hop_size in ms

    audio = np.pad(audio, int(fft_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - fft_size) / frame_len) + 1
    frames = np.zeros((frame_num, fft_size))

    for n in range(frame_num):
        frames[n] = audio[n * frame_len:n * frame_len + fft_size]

    return frames


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def met_to_freq(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, fft_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)

    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))

    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
    freqs = met_to_freq(mels)

    return np.floor((fft_size + 1) / sample_rate * freqs).astype(int), freqs


def get_filters(filter_points, fft_size):
    filters = np.zeros((len(filter_points) - 2, int(fft_size / 2 + 1)))

    for n in range(len(filter_points) - 2):
        filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[
            n + 1])

    return filters


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num, filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)

    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

    return basis
