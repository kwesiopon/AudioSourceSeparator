from Training import train

#!/usr/bin/python3
dataset_path = "/Users/Kwesi/PycharmProjects/AudioSourceSeparator/musdb18/train"

model_train = train(dataset_path, [100000, 1])

print(model_train)