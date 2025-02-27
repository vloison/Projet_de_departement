import numpy as np


def mono_label(Y_real, Y_pred):
    label_real = Y_real.argmax(axis=1)
    label_pred = Y_pred.argmax(axis=1)
    return (label_real == label_pred).mean()


def mono_label_KNN(y_real, y_pred):
    return(int(y_real.argmax() >0 and y_real.argmax() == y_pred.argmax()))


def multi_label(Y_real, Y_pred):
    return np.sum(np.logical_and(Y_real, Y_pred))/np.sum(np.logical_or(Y_real, Y_pred))


def notebook(y_test, pred):
    value = 0
    for i in range(0, len(pred)):
        first3_index = np.argsort(pred[i])[-3:]
        correct = np.where(y_test[i] == 1)[0]
        for j in first3_index:
            if j in correct:
                value += 1
    return value/len(pred)
