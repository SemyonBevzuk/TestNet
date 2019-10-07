from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt


def get_MNIST_Keras():
    num_train = 60000
    num_test = 10000
    height, width = 28, 28
    num_classes = 10
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(num_train, height * width)
    X_test = X_test.reshape(num_test, height * width)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)
    return (X_train, Y_train), (X_test, Y_test)


def normalized_MNIST(X_train, X_test):
    X_train /= 255
    X_test /= 255
    return X_train, X_test


def main():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    plt.imshow(X_train[0], cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
