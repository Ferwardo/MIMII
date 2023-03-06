import numpy
import numpy as np

import transforms
import os

from multiprocessing import Process


# Simple frequency plot of all channels
# transforms.frequencyPlot("00000001.wav", filepath="./fan/id_00/abnormal")

# Spectogram of all channels
# transforms.spectogram("00000001.wav", filepath="./fan/id_00/abnormal")

# MFCC of all channels
def abnormalID00(return_values):
    dirAudio = "./dataset/valve/id_00/abnormal"
    # dirAudio = "./temp/valve/id_00/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID00(return_values):
    dirAudio = "./dataset/valve/id_00/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID02(return_values):
    dirAudio = "./dataset/valve/id_02/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID02(return_values):
    dirAudio = "./dataset/valve/id_02/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID04(return_values):
    dirAudio = "./dataset/valve/id_04/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID04(return_values):
    dirAudio = "./dataset/valve/id_04/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID06(return_values):
    dirAudio = "./dataset/valve/id_06/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID06(return_values):
    dirAudio = "./dataset/valve/id_06/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


if __name__ == '__main__':
    mfcc1 = []
    mfcc2 = []
    mfcc3 = []
    mfcc4 = []
    mfcc5 = []
    mfcc6 = []
    mfcc7 = []
    mfcc8 = []

    process1 = Process(target=abnormalID00, args=(mfcc1,))
    process2 = Process(target=normalID00, args=(mfcc2,))
    process3 = Process(target=abnormalID02, args=(mfcc3,))
    process4 = Process(target=normalID02, args=(mfcc4,))
    process5 = Process(target=abnormalID04, args=(mfcc5,))
    process6 = Process(target=normalID04, args=(mfcc6,))
    process7 = Process(target=abnormalID06, args=(mfcc7,))
    process8 = Process(target=normalID06, args=(mfcc8,))

    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    process7.start()
    process8.start()

    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()
    process7.join()
    process8.join()

    print("All done with calculating mfcc. Now calculating mean and standard deviation")

    # concat all mfcc frames from each different device and class to each other
    mfcc1 = np.concatenate(np.concatenate(mfcc1))
    mfcc2 = np.concatenate(np.concatenate(mfcc1))
    mfcc3 = np.concatenate(np.concatenate(mfcc1))
    mfcc4 = np.concatenate(np.concatenate(mfcc1))
    mfcc5 = np.concatenate(np.concatenate(mfcc1))
    mfcc6 = np.concatenate(np.concatenate(mfcc1))
    mfcc7 = np.concatenate(np.concatenate(mfcc1))
    mfcc8 = np.concatenate(np.concatenate(mfcc1))

    all_mfccs = np.concatenate[mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8]

    mean = all_mfccs.mean()
    std = all_mfccs.std()

    np.save("./dataset/mean_std.npy", numpy.array([mean, std]))
