from tensorflow import keras


def create_resnet(nb_genres, size):
    resnet = keras.applications.resnet_v2.ResNet50V2(input_shape=size, include_top=False, weights="imagenet")
    resnet.trainable = False

    classifier = keras.models.Sequential([
        resnet,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(nb_genres, activation="softmax")
    ])

    classifier.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adagrad(),
        metrics=["accuracy"]
    )

    return classifier
