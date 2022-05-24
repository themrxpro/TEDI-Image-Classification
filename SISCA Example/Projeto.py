import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

target_epoch = 5

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


def trainNN(optimizer, n_neuron):
    with tf.device('/cpu:0'):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(n_neuron, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])

        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        labels = ["Epoch", "Loss", "Accuracy", "Real_Loss", "Real_Accuracy"]
        arr = np.array([])
        for this_epoch in range(1, target_epoch + 1):
            #a = model.fit(train_images, train_labels, epochs=1, batch_size=60000)
            #test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=10000)
            a = model.fit(train_images, train_labels, epochs=1)
            test_loss, test_acc = model.evaluate(test_images, test_labels)

            arr = np.append(arr, [this_epoch,
                                  a.history["loss"][0],
                                  a.history["accuracy"][0],
                                  test_loss,
                                  test_acc], axis=0)
        arr = arr.reshape(target_epoch, len(labels))

    return pd.DataFrame(data=arr, columns=labels)


def save_data(data, name):
    with open(f"{name}.pkl", 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def load_data(name):
    with open(f"{name}.pkl", 'rb') as pickle_file:
        return pickle.load(pickle_file)


###
def test_n_neurons():
    # Test number of neurons
    results = []
    neuron_runs = [2, 4, 16, 32, 64, 128, 256, 512]
    ###
    for n_neuron in neuron_runs:
        start_time = time.time()
        NN_info = trainNN("adam", n_neuron)
        run_time = time.time() - start_time

        results.append([n_neuron, NN_info, run_time])
    ###

def test_optimizers():
    optimizers = ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop", "SGD"]

    results = []
    for optimizer in optimizers:
        results.append(trainNN(optimizer, 256))

    for i in range(len(optimizers)):
        plt.plot(results[i]["Epoch"], results[i]["Real_Accuracy"])
    plt.legend(optimizers)
    plt.xlabel("Epochs")
    plt.ylabel("Precisão")
    plt.title("Precisão ao longo dos Epochs")
    plt.show()
