import transforms
import os

# Simple frequency plot of all channels
# transforms.frequencyPlot("00000001.wav", filepath="./fan/id_00/abnormal")

# Spectogram of all channels
# transforms.spectogram("00000001.wav", filepath="./fan/id_00/abnormal")

# MFCC of all channels
dirAudio = "./dataset/valve/id_00/abnormal"
for file in os.listdir(dirAudio):
    if file == "mfcc":
        continue
    print(f"Working on: {dirAudio}/{file}")
    mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)

dirAudio = "./dataset/valve/id_00/normal"
for file in os.listdir(dirAudio):
    if file == "mfcc":
        continue
    print(f"Working on:{dirAudio}/{file}")
    mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)

dirAudio = "./dataset/valve/id_02/abnormal"
for file in os.listdir(dirAudio):
    if file == "mfcc":
        continue
    print(f"Working on: {dirAudio}/{file}")
    mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)

dirAudio = "./dataset/valve/id_02/normal"
for file in os.listdir(dirAudio):
    if file == "mfcc":
        continue
    print(f"Working on:{dirAudio}/{file}")
    mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)

dirAudio = "./dataset/valve/id_04/abnormal"
for file in os.listdir(dirAudio):
    if file == "mfcc":
        continue
    print(f"Working on: {dirAudio}/{file}")
    mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)

dirAudio = "./dataset/valve/id_04/normal"
for file in os.listdir(dirAudio):
    if file == "mfcc":
        continue
    print(f"Working on:{dirAudio}/{file}")
    mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)

dirAudio = "./dataset/valve/id_06/abnormal"
for file in os.listdir(dirAudio):
    if file == "mfcc":
        continue
    print(f"Working on: {dirAudio}/{file}")
    mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)

dirAudio = "./dataset/valve/id_06/normal"
for file in os.listdir(dirAudio):
    if file == "mfcc":
        continue
    print(f"Working on:{dirAudio}/{file}")
    mfcc = transforms.MFCC(file, filepath=dirAudio, saveMFCC=True)
