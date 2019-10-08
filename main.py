import argparse
import my_net
import keras_net


def create_parser():
    parser = argparse.ArgumentParser(
        description='Реализация полносвязной двухслойной нейронной сети для задачи классификации данных MNIST.')
    parser.add_argument('--net_type', type=str, choices=['my', 'keras'], default='my',
                        help='Выбор сети для запуска. Собственная - my, фреймворк - keras')
    parser.add_argument('--hidden_size', type=int, default=30,
                        help='Число узлов на скрытом слое.')
    parser.add_argument('--lr_hidden', type=float, default=0.1,
                        help='Скорость обучения на скрытом слое (только для собственной реализации) [0, 1].')
    parser.add_argument('--lr_output', type=float, default=0.1,
                        help='Скорость обучения на выходном слое или всей сети для keras.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Размер пакета для обучения.')
    parser.add_argument('--number_epochs', type=int, default=20,
                        help='Число эпох для обучения.')
    parser.add_argument('--compare_nets', action='store_true',
                        help='Флаг для запуска сравнения двух реализаций.')
    return parser


def print_parameters(args):
    print('\n\tParameters')
    print('hidden_size = ', args.hidden_size)
    print('lr_hidden = ', args.lr_hidden)
    print('lr_output = ', args.lr_output)
    print('batch_size =', args.batch_size)
    print('number_epochs = ', args.number_epochs)
    print()


def run_my_net(args):
    score_train, score_test, delta_time = my_net.fit_and_test_net_on_MNIST(
        args.hidden_size, args.batch_size, args.number_epochs, args.lr_hidden, args.lr_output)
    return score_train, score_test, delta_time


def run_keras_net(args):
    score_train, score_test, delta_time = keras_net.fit_and_test_net_on_MNIST(
        args.hidden_size, args.batch_size, args.number_epochs, args.lr_hidden)
    return score_train, score_test, delta_time


def print_results(score_train, score_test, delta_time):
    print()
    print('Delta time =', delta_time)
    print('Train loss:', score_train[0])
    print('Train accuracy:', score_train[1])
    print('Test loss:', score_test[0])
    print('Test accuracy:', score_test[1])
    print()


def main(args):
    print_parameters(args)
    if args.compare_nets:
        print('\tComparison mode.')
        my_score_train, my_score_test, my_delta_time = run_my_net(args)
        keras_score_train, keras_score_test, keras_delta_time = run_keras_net(args)
        print()

        print('\tMy implementation:')
        print_results(my_score_train, my_score_test, my_delta_time)

        print('\tKeras implementation:')
        print_results(keras_score_train, keras_score_test, keras_delta_time)
    else:
        if args.net_type == 'my':
            print('\tMy implementation:')
            score_train, score_test, delta_time = run_my_net(args)
            print_results(score_train, score_test, delta_time)
        elif args.net_type == 'keras':
            print('\tKeras implementation:')
            score_train, score_test, delta_time = run_keras_net(args)
            print_results(score_train, score_test, delta_time)
        else:
            print('Ошибка в параметре --net_type: my, keras')


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())

'''
python main.py --net_type my --hidden_size 30 --lr_hidden 0.1 --lr_output -0.1 --batch_size 128 --number_epochs 20 --compare_nets
'''
