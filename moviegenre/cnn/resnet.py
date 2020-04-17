from tensorflow import keras, identity
from tensorflow.keras.applications.resnet_v2 import preprocess_input


def create_resnet(nb_genres, size):
    resnet = keras.applications.resnet_v2.ResNet50V2(input_shape=size, include_top=False, weights="imagenet")
    resnet.trainable = False

    model = keras.models.Sequential([
        resnet,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(nb_genres, activation="softmax")
    ])

    i = keras.layers.Input(shape=size)
    x = identity(i)
    #x = preprocess_input(x)
    output = model(x)

    classifier = keras.models.Model(inputs=i, outputs=output)

    classifier.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adagrad(),
        metrics=["accuracy"]
    )

    return classifier
