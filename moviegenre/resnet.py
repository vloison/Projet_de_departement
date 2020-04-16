from tensorflow import keras

resnet = keras.applications.resnet_v2.ResNet50V2(input_shape=(50, 50, 3), include_top=False, weights="imagenet")

resnet.trainable = False
