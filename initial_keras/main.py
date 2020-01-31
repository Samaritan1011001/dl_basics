from __future__ import absolute_import, division, print_function, unicode_literals
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.datasets import mnist
from keras.datasets import imdb
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf


def main():
    # rec_digits()

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.executing_eagerly())
    class_revs()


def class_revs():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print(train_data[0])
    print(train_labels[0])
    print(max([max(seq) for seq in train_data]))

    word_index = imdb.get_word_index()
    # print(word_index)
    reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
    # print(reverse_word_index)
    decode_review = " ".join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decode_review)

    # Prepare the data
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # # Create validation set
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    print(x_train[0])

    # # Build the network
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    #
    # # Compile the model
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])
    history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
    history_dict = history.history
    print(history_dict)

    return


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def rec_digits():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_images.shape)
    print(len(train_labels))
    print(test_images.shape)
    print(len(test_labels))
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc: ', test_acc)
    return


if __name__ == "__main__":
    main()
