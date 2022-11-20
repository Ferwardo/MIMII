import transforms

# Simple frequency plot of all channels
# transforms.frequencyPlot("00000001.wav", filepath="./fan/id_00/abnormal")

# Spectogram of all channels
# transforms.spectogram("00000001.wav", filepath="./fan/id_00/abnormal")

# MFCC of all channels
transforms.MFCC("00000001.wav", filepath="fan/id_00/abnormal")
