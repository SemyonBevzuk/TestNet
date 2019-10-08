from keras.datasets import mnist
from keras.utils import np_utils


def get_MNIST_Keras():
    num_categories = 10
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = np_utils.to_categorical(Y_train, num_categories)
    Y_test = np_utils.to_categorical(Y_test, num_categories)
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
