import numpy as np


def mono_label(Y_real, Y_pred):
    label_real = Y_real.argmax(axis=1)
    label_pred = Y_pred.argmax(axis=1)

    return (label_real == label_pred).mean()
