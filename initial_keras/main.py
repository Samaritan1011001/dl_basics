from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.datasets import mnist
from keras.datasets import imdb
from keras.datasets import reuters
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy

def main():
    # rec_digits()

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # print(tf.executing_eagerly())
    # class_revs()
    news_wires()

def random_classifier(test_labels):
    test_labels_copy = copy.copy(test_labels)
    np.random.shuffle(test_labels_copy)
    hits_array = np.array(test_labels) == np.array(test_labels_copy)
    print(f'Baseline accuracy for random testt labels: {float(np.sum(hits_array))/len(test_labels)}')
    return

def news_wires():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    print(train_data[0])
    print(train_labels[0])
    random_classifier(test_labels)

    word_index = reuters.get_word_index()
    # print(word_index)
    reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
    # print(reverse_word_index)
    decode_review = " ".join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decode_review)

    # Prepare the data
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # # Create validation set
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]
    print(x_train[0])
    # one_hot_train_labels = to_one_hot(train_labels)
    # one_hot_test_labels = to_one_hot(test_labels)

    # # Build the network
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    # Hidden layers cannot be too small because if it is, then there will be information loss
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    # Compile the model
    # Use sparse categorical cross entropy for loss if NOT using one hot encoding. Both give same results
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # Training and validation
    history = model.fit(partial_x_train, partial_y_train, epochs=12, batch_size=512, validation_data=(x_val, y_val))
    history_dict = history.history
    print(history_dict.keys())
    # plot_epocs_graph(history_dict,False)
    results = model.evaluate(x_test, one_hot_test_labels)
    print(results)
    return


def to_one_hot(labels, dimensions=48):
    results = np.zeros((len(labels), dimensions))
    for i, sequence in enumerate(labels):
        results[i, sequence] = 1.
    return results


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
    history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))
    history_dict = history.history
    print(history_dict.keys())
    plot_epocs_graph(history_dict)
    results = model.evaluate(x_test, y_test)
    print(results)

    return


def plot_epocs_graph(history_dict,bin_acc=True):
    loss_vals = history_dict['loss']
    val_loss_vals = history_dict['val_loss']
    epochs = range(1, len(history_dict['binary_accuracy']if bin_acc else history_dict['acc']) + 1)
    plt.plot(epochs, loss_vals, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss_vals, 'b', label='Validation Loss')
    plt.title("Training and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc_vals = history_dict['binary_accuracy']if bin_acc else history_dict['acc']
    val_acc_vals = history_dict['val_binary_accuracy']if bin_acc else history_dict['val_acc']
    plt.plot(epochs, acc_vals, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc_vals, 'b', label='Validation accuracy')
    plt.title("Training and validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
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
