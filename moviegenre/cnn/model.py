# -*- coding: utf-8 -*-
from tensorflow import keras, identity


def standard_layer(conv1_dim, conv2_dim, input):
    output = keras.layers.Conv2D(conv1_dim, kernel_size=(3, 3), activation="relu")(input)
    output = keras.layers.Conv2D(conv2_dim, kernel_size=(3, 3), activation="relu")(output)
    output = keras.layers.MaxPooling2D(pool_size=(2, 2))(output)
    output = keras.layers.Dropout(0.25)(output)

    return output


def create_cnn(nb_genres, size):
    input_poster = keras.layers.Input(shape=size)

    output = standard_layer(32, 64, input_poster)

    output = standard_layer(128, 64, output)

    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(128, activation="relu")(output)
    output = keras.layers.Dropout(0.5)(output)
    output = keras.layers.Dense(nb_genres, activation="sigmoid")(output)

    model = keras.models.Model(inputs=[input_poster], outputs=output)

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adagrad(),
        metrics=["accuracy", keras.metrics.categorical_accuracy]
    )

    return model


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
    #x = keras.applications.resnet_v2.preprocess_input(x)
    output = model(x)

    classifier = keras.models.Model(inputs=i, outputs=output)

    classifier.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adagrad(),
        metrics=["accuracy"]
    )

    return classifier
