import numpy
import numpy as np

import transforms
import os

from multiprocessing import Process


# Simple frequency plot of all channels
# transforms.frequencyPlot("00000001.wav", filepath="./fan/id_00/abnormal")

# Spectogram of all channels
# transforms.spectogram("00000001.wav", filepath="./fan/id_00/abnormal")

# MFCC of all channels: Fan
def abnormalID00(return_values):
    dirAudio = "./dataset/fan/id_00/abnormal"
    # dirAudio = "./temp/valve/id_00/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID00(return_values):
    dirAudio = "./dataset/fan/id_00/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID02(return_values):
    dirAudio = "./dataset/fan/id_02/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID02(return_values):
    dirAudio = "./dataset/fan/id_02/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID04(return_values):
    dirAudio = "./dataset/fan/id_04/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID04(return_values):
    dirAudio = "./dataset/fan/id_04/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID06(return_values):
    dirAudio = "./dataset/fan/id_06/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID06(return_values):
    dirAudio = "./dataset/fan/id_06/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


# MFCC of all channels: Pump
def abnormalID00_pump(return_values):
    dirAudio = "./dataset/pump/id_00/abnormal"
    # dirAudio = "./temp/valve/id_00/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID00_pump(return_values):
    dirAudio = "./dataset/pump/id_00/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID02_pump(return_values):
    dirAudio = "./dataset/pump/id_02/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID02_pump(return_values):
    dirAudio = "./dataset/pump/id_02/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID04_pump(return_values):
    dirAudio = "./dataset/pump/id_04/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID04_pump(return_values):
    dirAudio = "./dataset/pump/id_04/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID06_pump(return_values):
    dirAudio = "./dataset/pump/id_06/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID06_pump(return_values):
    dirAudio = "./dataset/pump/id_06/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


# MFCC of all channels: Valve
def abnormalID00_valve(return_values):
    dirAudio = "./dataset/valve/id_00/abnormal"
    # dirAudio = "./temp/valve/id_00/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID00_valve(return_values):
    dirAudio = "./dataset/valve/id_00/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID02_valve(return_values):
    dirAudio = "./dataset/valve/id_02/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID02_valve(return_values):
    dirAudio = "./dataset/valve/id_02/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID04_valve(return_values):
    dirAudio = "./dataset/valve/id_04/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID04_valve(return_values):
    dirAudio = "./dataset/valve/id_04/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID06_valve(return_values):
    dirAudio = "./dataset/valve/id_06/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID06_valve(return_values):
    dirAudio = "./dataset/valve/id_06/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


# MFCC of all channels: Slider
def abnormalID00_slider(return_values):
    dirAudio = "./dataset/slider/id_00/abnormal"
    # dirAudio = "./temp/valve/id_00/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID00_slider(return_values):
    dirAudio = "./dataset/slider/id_00/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID02_slider(return_values):
    dirAudio = "./dataset/slider/id_02/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID02_slider(return_values):
    dirAudio = "./dataset/slider/id_02/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID04_slider(return_values):
    dirAudio = "./dataset/slider/id_04/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID04_slider(return_values):
    dirAudio = "./dataset/slider/id_04/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def abnormalID06_slider(return_values):
    dirAudio = "./dataset/slider/id_06/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on: {dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


def normalID06_slider(return_values):
    dirAudio = "./dataset/slider/id_06/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel" or file.endswith(".npy"):
            continue
        print(f"Working on:{dirAudio}/{file}")
        return_values.append(transforms.MFCC(file, filepath=dirAudio, saveMFCC=True))


redo_dataset = False

if __name__ == '__main__':
    if redo_dataset:
        mfcc1 = []
        mfcc2 = []
        mfcc3 = []
        mfcc4 = []
        mfcc5 = []
        mfcc6 = []
        mfcc7 = []
        mfcc8 = []

        mfcc1_pump = []
        mfcc2_pump = []
        mfcc3_pump = []
        mfcc4_pump = []
        mfcc5_pump = []
        mfcc6_pump = []
        mfcc7_pump = []
        mfcc8_pump = []

        mfcc1_valve = []
        mfcc2_valve = []
        mfcc3_valve = []
        mfcc4_valve = []
        mfcc5_valve = []
        mfcc6_valve = []
        mfcc7_valve = []
        mfcc8_valve = []

        mfcc1_slider = []
        mfcc2_slider = []
        mfcc3_slider = []
        mfcc4_slider = []
        mfcc5_slider = []
        mfcc6_slider = []
        mfcc7_slider = []
        mfcc8_slider = []

        process1 = Process(target=abnormalID00, args=(mfcc1,))
        process2 = Process(target=normalID00, args=(mfcc2,))
        process3 = Process(target=abnormalID02, args=(mfcc3,))
        process4 = Process(target=normalID02, args=(mfcc4,))
        process5 = Process(target=abnormalID04, args=(mfcc5,))
        process6 = Process(target=normalID04, args=(mfcc6,))
        process7 = Process(target=abnormalID06, args=(mfcc7,))
        process8 = Process(target=normalID06, args=(mfcc8,))

        process1_pump = Process(target=abnormalID00_pump, args=(mfcc1_pump,))
        process2_pump = Process(target=normalID00_pump, args=(mfcc2_pump,))
        process3_pump = Process(target=abnormalID02_pump, args=(mfcc3_pump,))
        process4_pump = Process(target=normalID02_pump, args=(mfcc4_pump,))
        process5_pump = Process(target=abnormalID04_pump, args=(mfcc5_pump,))
        process6_pump = Process(target=normalID04_pump, args=(mfcc6_pump,))
        process7_pump = Process(target=abnormalID06_pump, args=(mfcc7_pump,))
        process8_pump = Process(target=normalID06_pump, args=(mfcc8_pump,))

        process1_valve = Process(target=abnormalID00_valve, args=(mfcc1_valve,))
        process2_valve = Process(target=normalID00_valve, args=(mfcc2_valve,))
        process3_valve = Process(target=abnormalID02_valve, args=(mfcc3_valve,))
        process4_valve = Process(target=normalID02_valve, args=(mfcc4_valve,))
        process5_valve = Process(target=abnormalID04_valve, args=(mfcc5_valve,))
        process6_valve = Process(target=normalID04_valve, args=(mfcc6_valve,))
        process7_valve = Process(target=abnormalID06_valve, args=(mfcc7_valve,))
        process8_valve = Process(target=normalID06_valve, args=(mfcc8_valve,))

        process1_slider = Process(target=abnormalID00_slider, args=(mfcc1_slider,))
        process2_slider = Process(target=normalID00_slider, args=(mfcc2_slider,))
        process3_slider = Process(target=abnormalID02_slider, args=(mfcc3_slider,))
        process4_slider = Process(target=normalID02_slider, args=(mfcc4_slider,))
        process5_slider = Process(target=abnormalID04_slider, args=(mfcc5_slider,))
        process6_slider = Process(target=normalID04_slider, args=(mfcc6_slider,))
        process7_slider = Process(target=abnormalID06_slider, args=(mfcc7_slider,))
        process8_slider = Process(target=normalID06_slider, args=(mfcc8_slider,))

        process1.start()
        process2.start()
        process3.start()
        process4.start()
        process5.start()
        process6.start()
        process7.start()
        process8.start()

        process1_pump.start()
        process2_pump.start()
        process3_pump.start()
        process4_pump.start()
        process5_pump.start()
        process6_pump.start()
        process7_pump.start()
        process8_pump.start()

        process1_valve.start()
        process2_valve.start()
        process3_valve.start()
        process4_valve.start()
        process5_valve.start()
        process6_valve.start()
        process7_valve.start()
        process8_valve.start()

        process1_slider.start()
        process2_slider.start()
        process3_slider.start()
        process4_slider.start()
        process5_slider.start()
        process6_slider.start()
        process7_slider.start()
        process8_slider.start()

        process1.join()
        process2.join()
        process3.join()
        process4.join()
        process5.join()
        process6.join()
        process7.join()
        process8.join()

        process1_pump.join()
        process2_pump.join()
        process3_pump.join()
        process4_pump.join()
        process5_pump.join()
        process6_pump.join()
        process7_pump.join()
        process8_pump.join()

        process1_valve.join()
        process2_valve.join()
        process3_valve.join()
        process4_valve.join()
        process5_valve.join()
        process6_valve.join()
        process7_valve.join()
        process8_valve.join()

        process1_slider.join()
        process2_slider.join()
        process3_slider.join()
        process4_slider.join()
        process5_slider.join()
        process6_slider.join()
        process7_slider.join()
        process8_slider.join()

        print("All done with calculating mfcc. Now calculating mean and standard deviation")

    all_mfccs = []

    for device_type in os.listdir("./dataset"):
        if device_type == "mean_std.npy":
            continue
        for id in os.listdir(f"./dataset/{device_type}"):
            for data_class in os.listdir(f"./dataset/{device_type}/{id}"):
                for file in os.listdir(f"./dataset/{device_type}/{id}/{data_class}"):
                    if file == "mfcc" or file == "logmel" or file.endswith(".wav"):
                        continue
                    # Save the mean and std of each mfcc to disk
                    all_mfccs.append(np.concatenate(
                        np.load(f"./dataset/{device_type}/{id}/{data_class}/{file}", allow_pickle=True).astype(
                            "float32")))

    all_mfccs = np.concatenate(np.array(all_mfccs))
    mean = all_mfccs.mean()
    std = all_mfccs.std()

    np.save("./dataset_means_std/means_std.npy", numpy.array([mean, std]))

    for device_type in os.listdir("./dataset"):
        if device_type == "mean_std.npy":
            continue
        for id in os.listdir(f"./dataset/{device_type}"):
            for data_class in os.listdir(f"./dataset/{device_type}/{id}"):
                for file in os.listdir(f"./dataset/{device_type}/{id}/{data_class}"):
                    if file == "mfcc" or file == "logmel" or file.endswith(".wav"):
                        continue
                    # Save the mean and std of each mfcc to disk
                    np_file = np.load(f"./dataset/{device_type}/{id}/{data_class}/{file}", allow_pickle=True).astype(
                        "float32")
                    # normalise first
                    np_file = (np_file - mean) / std

                    means_stds = []

                    for channel in range(0, np_file.shape[0]):
                        means = []
                        stds = []
                        for filter in range(0, np_file.shape[2]):
                            means.append(np_file[channel, filter].mean())
                            stds.append(np_file[channel, filter].std())
                        means_stds.append(means + stds)

                    os.makedirs(f"./dataset_mean_std/{device_type}/{id}/{data_class}/", exist_ok=True)
                    np.save(f"./dataset_mean_std/{device_type}/{id}/{data_class}/{file}", np.asarray(means_stds))
