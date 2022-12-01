import transforms
import os

from multiprocessing import Process


# Simple frequency plot of all channels
# transforms.frequencyPlot("00000001.wav", filepath="./fan/id_00/abnormal")

# Spectogram of all channels
# transforms.spectogram("00000001.wav", filepath="./fan/id_00/abnormal")

# MFCC of all channels
def abnormalID00():
    dirAudio = "./dataset/valve/id_00/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on: {dirAudio}/{file}")
        mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)


def normalID00():
    dirAudio = "./dataset/valve/id_00/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on:{dirAudio}/{file}")
        mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)


def abnormalID02():
    dirAudio = "./dataset/valve/id_02/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on: {dirAudio}/{file}")
        mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)


def normalID02():
    dirAudio = "./dataset/valve/id_02/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on:{dirAudio}/{file}")
        mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)


def abnormalID04():
    dirAudio = "./dataset/valve/id_04/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on: {dirAudio}/{file}")
        mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)


def normalID04():
    dirAudio = "./dataset/valve/id_04/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on:{dirAudio}/{file}")
        mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)


def abnormalID06():
    dirAudio = "./dataset/valve/id_06/abnormal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on: {dirAudio}/{file}")
        mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)


def normalID06():
    dirAudio = "./dataset/valve/id_06/normal"
    for file in os.listdir(dirAudio):
        if file == "mfcc" or file == "logmel":
            continue
        print(f"Working on:{dirAudio}/{file}")
        mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)


if __name__ == '__main__':
    process1 = Process(target=abnormalID00)
    process2 = Process(target=normalID00)
    process3 = Process(target=abnormalID02)
    process4 = Process(target=normalID02)
    process5 = Process(target=abnormalID04)
    process6 = Process(target=normalID04)
    process7 = Process(target=abnormalID06)
    process8 = Process(target=normalID06)

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

    print("All Done")
