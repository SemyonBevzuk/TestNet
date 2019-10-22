from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense
from datetime import datetime
import read_mnist


def fit_and_test_net_on_MNIST(hidden_size=30, batch_size=128, num_epochs=20, lr=0.1):
    (X_train, Y_train), (X_test, Y_test) = read_mnist.get_MNIST_Keras()
    X_train, X_test = read_mnist.normalized_MNIST(X_train, X_test)

    time_start = datetime.now()
    inp = Input(shape=(X_train.shape[1],))  # Our input is a 1D vector of size 784
    hidden_1 = Dense(hidden_size, activation='relu')(inp)  # First hidden ReLU layer
    out = Dense(Y_train.shape[1], activation='softmax')(hidden_1)  # Output softmax layer
    model = Model(input=inp, output=out)  # To define a model, just specify its input and output layers
    sgd = optimizers.SGD(lr=lr, momentum=0.0, nesterov=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=num_epochs,
              verbose=2)

    delta_time = datetime.now() - time_start
    score_train = model.evaluate(X_train, Y_train, verbose=0)
    score_test = model.evaluate(X_test, Y_test, verbose=0)
    return score_train, score_test, delta_time


if __name__ == '__main__':
    score_train, score_test, delta_time = fit_and_test_net_on_MNIST(300, 128, 20, 0.1)
    print()
    print('Delta time =', delta_time)
    print('Test loss:', score_test[0])
    print('Test accuracy:', score_test[1])
