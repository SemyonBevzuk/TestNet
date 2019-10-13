# Лабораторная работа №1
Реализована двуслойная полносвязная нейронная сеть
для распознования цифр из базы MNIST.    

    Входной слой - изображение из базы MNIST
    Скрытый слой - функция активации ReLU
    Выходной слой - функция активации Softmax
    Функция ошибки - кросс-энтропия
    
## Математическая модель нейрона
Математическая модель нейрона имеет следующий вид:
![](https://latex.codecogs.com/svg.latex?u_k=b_k&plus;\sum\limits_{j=1}^nw_{k,j}x_j\qquad&space;y_k=\phi(u_k))

Где ![](https://latex.codecogs.com/svg.latex?\phi) - функция активации, ![](https://latex.codecogs.com/svg.latex?b_k) - 
смещение, ![](https://latex.codecogs.com/svg.latex?w_{k,j}) - вес, ![](https://latex.codecogs.com/svg.latex?x) - вход.

Для удобства выкладок сделаем некоторое преобразование. Внесем смещение в сумму с новым значением синапса ![](https://latex.codecogs.com/svg.latex?x_0=1).
Тогда модель нейрона можно запсиать в следующем виде:

![](https://latex.codecogs.com/svg.latex?u_k=\sum\limits_{j=0}^nw_{k,j}x_j\qquad&space;y_k=\phi(u_k))

## Предобработка данных
Входные данные нормируются, представляются как матрицы и вектора.

## Начальная инициализация весов
Инициализация весов осуществляется с использованием метода Ксавье.

![](https://latex.codecogs.com/svg.latex?W=\sigma*N(0,1)\qquad&space;\sigma=2/\sqrt{size_{input}+size_{output}})

## Функции активации
### На скрытом слое
На скрытом слое будем использовать ReLU:

![](https://latex.codecogs.com/svg.latex?\phi^{(1)}(u)=max(0,u))

### На выходном слое
На выходе будем использовать функцию Softmax:

![](https://latex.codecogs.com/svg.latex?\phi^{(2)}(u_j)=\frac{e^{u_j}}{\sum\limits_{i=0}^ne^{u_i}})

Её производные:

![](https://latex.codecogs.com/svg.latex?\frac{\partial\phi^{(2)}(u_j)}{\partial u_j}=\phi^{(2)}(u_j)(1-\phi^{(2)}(u_j)))

![](https://latex.codecogs.com/svg.latex?\frac{\partial\phi^{(2)}(u_j)}{\partial u_i}=-\phi^{(2)}(u_j)\phi^{(2)}(u_i))

## Функция ошибки
В качестве функции ошибки рассмотрим кросс-энтропию:
![](https://latex.codecogs.com/svg.latex?E=\sum\limits_{j=1}^M y_j\ln{u_j})

Где *y* - выход сети, *u* - ожидаемый выход. 


$'\sqrt{2}'$

![](https://latex.codecogs.com/svg.latex?2)

![](https://latex.codecogs.com/svg.latex?\sqrt{2})