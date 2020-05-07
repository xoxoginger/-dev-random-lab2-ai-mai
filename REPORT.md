# Отчет по лабораторной работе 
## по курсу "Искусственый интеллект"

## Нейросетям для распознавания изображений


### Студенты: 

| ФИО       | Роль в проекте                     | Оценка       |
|-----------|------------------------------------|--------------|
| [Алексюнина Ю.В.](https://github.com/xoxoginger) | Подготовка датасета, обучение полносвязной нейронной сети, напичание отчёта  |          |
| [Бондарь К.Г.](https://github.com/ksenbond) | Подготовка датасета, обучение свёрточной нейронной сети, написание отчёта |       |
| [Лопатин А.О.](https://github.com/alexlopatin) | Подготовка датасета, обучение полносвязной нейронной сети, обучение свёрточной нейронной сети |      |

## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В. |              |               |

> *Комментарии проверяющих (обратите внимание, что более подробные комментарии возможны непосредственно в репозитории по тексту программы)*

## Тема работы

Опишите тему работы, включая предназначенный для распознавания набор символов.

Подготовка набора данных и построение нескольких нейросетевых классификаторов для распознавания рукописных символов. 

Вариант:
А - 192
Б - 193
Л - 203

(192+193+203)%5+1=4 - Значки, использующиеся при голосовании: checkmark и крестик

### Распределение работы в команде

Каждый участник написал по набору с данными. Тестовый и обучающий датасет были сформирован случайным образом: наугад был выбран третий набор  как тестовый, а остальные два как обучающие. Скрипт для разрезания изображений написала Алексюнина Юлия.
После этого обсуждался принцип работы нейронных сетей и фреймворка Keras.


## Подготовка данных

### Фотографии исходных листков с рукописными символами:


[Набор 1](cropping/1_final.jpg)

[Набор 2](cropping/2_final.jpg)

[Набор 3](cropping/3_final.jpg)


Подготовка датасета осуществлялась следующим образом:

Сначала мы написали от руки три набора данных по 200 символов в каждом. После отсканировали их при помощи программы Microsoft Office Lens в режиме 'документ'. 

Затем попробовали написать скрипт на Python, чтобы разрезать изображения. Использовали opencv и при помощи статьи на tproger.ru разобрались с этим.

Поскольку мы не учли, что изображение все же под некоторым(и, как оказалось, довольно большим) углом, мы попытались сделать изображение под минимальным углом (с помощью штатива и линейки), а также нашли специальную функцию в Microsoft Office Lens, которая помогла нам в этом.

После этого нам удалось получилось 600 изображений, по 200 на каждый набор.

Ссылка на получившийся датасет: 

[data](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/tree/master/cropping/dataset)

Примеры разрезанных изображений:

Набор 1: ![11](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/cropping/dataset/data1/100.png)
![12](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/cropping/dataset/data1/220.png)

Набор 2: ![21](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/cropping/dataset/data2/304.png)
![22](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/cropping/dataset/data2/435.png)

Набор 3: ![31](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/cropping/dataset/data3/509.png)
![32](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/cropping/dataset/data3/639.png)

Основной функционал разрезания: 

```python
 while cnt_h < 20:  # кол-во по вертикали
        cnt_w = 0
        if x0 == 3:
            x0 = 3  # для 1
            x1 = 31
        else:
            x0 = 0 # 2
            x1 = 28
        while cnt_w < 10:  # кол-во по горизонтали
            cr_im = image[y0:y1, x0:x1]
            filename = '{0}.png'.format(cr_name)
            cv2.imwrite(filename, cr_im)
            cv2.imshow("Cropped image", cr_im)  # чтобы отсмотреть полученное
            cv2.waitKey(0)
            cnt_w += 1
            x0 += cr_width * 2
            x1 += cr_width * 2
            cr_name += 1
        cnt_h += 1
        y0 += cr_height * 2
        y1 += cr_height * 2
```

Исходный код: [data.py](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/cropping/data.py)

Первоначально размер каждого изображения был 74х74, но для увеличения скорости обучения мы уменьшили размеры получившихся изображений до 28х28

## Загрузка данных
```python
trainingImages = np.empty([400, 784])
trainingLabels = np.empty([400])

testImages = np.empty([200, 784])
testLabels = np.empty([200])

for i in range(0,99):
  trainingLabels[i] = 0
for i in range(100,199):
  trainingLabels[i] = 1
for i in range(200,299):
  trainingLabels[i] = 0
for i in range(300,399):
  trainingLabels[i] = 1

for i in range(0,199):
  testLabels[i] = int(i/100)

# cropping/dataset/data1 100
for i in range(100, 299):
  image = cv2.imread("cropping/dataset/data1/" + str(i) +".png")
  arr = np.asarray(image).reshape(-1)[::3]
  trainingImages[i-100] = arr
for i in range(300, 499):
  image = cv2.imread("cropping/dataset/data2/" + str(i) +".png")
  arr = np.asarray(image).reshape(-1)[::3]
  trainingImages[i-100] = arr
for i in range(500, 699):
  image = cv2.imread("cropping/dataset/data3/" + str(i) +".png")
  arr = np.asarray(image).reshape(-1)[::3]
  testImages[i-500] = arr
```

## Обучение нейросети

Мы используем модель последовательного типа. Этот метод позволяет выстраивать модель послойно. Для добавления слоев используем функцию  add().

В случае полносвязной сети(и однослойной, и многослойной) мы взяли количество нейронов на слой, равное 1024.

Метод DropOut помогает сети не переобучиться.

Softmax – это функция активации. Она сводит получившуюся сумму к 1, чтобы результат мог интерпретироваться как ряд возможных исходов. Тогда модель будет делать прогноз на основании того, какой вариант наиболее вероятен.

SGD - Оптимизатор - это один из аргументов, необходимых для компиляции модели Keras

Метод compile собирает модель сети
    
    loss='categorical_crossentropy' — Функция потерь.
    optimizer=sgd — Оптимизатор весов .

fit — функция для обучения, принимающая данные обучения, целевые данные, количество эпох и количество выборок для нейронной сети за раз.

Метод evaluate получает на вход тестовую выборку
verbose отвечает за то, как будет выглядеть вывод прогресса обучения каждой эпохи.
  verbose = 0 — не будет ничего показывать;
  verbose = 1 — строка с потерями, точностью;
  verbose = 2 — вывод номера эпохи;

### Полносвязная однослойная сеть

Полносвязная сеть - это значит, что каждый нейрон связан со всеми нейронами предыдущего слоя. 
Однослойные нейронные сети состоят из одного вычислительного слоя нейронов. Входной слой подает сигналы сразу на выходной слой, который и преобразует сигнал, и сразу выдает результат.

x_train — данные обучения
y_train — целевые данные

Данные обучения нормируются, чтобы избежать перенасыщения в функциях активации.
```python
x_train = trainingImages/255
y_train = keras.utils.to_categorical(trainingLabels, num_classes=2)
x_test = testImages/255
y_test = keras.utils.to_categorical(testLabels, num_classes=2)

model = Sequential()
model.add(Dense(1024, activation=keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None), input_dim=784))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
```

Исходный код: [learnNN.py](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/learnNN.py)

**Результаты**

```python
Train on 400 samples, validate on 200 samples

Epoch 20/20
400/400 [==============================] - 0s 347us/step - loss: 0.0334 - accuracy: 0.9875
200/200 [==============================] - 0s 220us/step
[0.14073900103569031, 0.949999988079071]
```
Точность: 94.99% 

### Полносвязная многослойная сеть

Многослойные нейронные сети, помимо входного и выходного слоев, имеют еще и скрытые слои. Эти скрытые слои проводят какие-то внутренние промежуточные преобразования, наподобие этапов производства продуктов на заводе.


```python
x_train = trainingImages/255
y_train = keras.utils.to_categorical(trainingLabels, num_classes=2)
x_test = testImages/255
y_test = keras.utils.to_categorical(testLabels, num_classes=2)

model = Sequential()
model.add(Dense(1024, activation=keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None), input_dim=784))
model.add(Dropout(0.2))
model.add(Dense(1024, activation=keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
```

Исходный код: [learnNN2layers.py](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/learnNN2layers.py)

**Результаты**

```python
Train on 400 samples, validate on 200 samples

Epoch 20/20
400/400 [==============================] - 0s 365us/step - loss: 0.0202 - accuracy: 0.9925
200/200 [==============================] - 0s 225us/step
[0.11129201114177704, 0.9549999833106995]
```
Точность: 95.49%

 В ходе наблюдений мы заметили, что бо́льшему количеству нейронов соответствовала бо́льшая точноть. 
 Так же стоит отметить, что полносвязная многослойная оказалась точнее однослойной 
 
### Свёрточная сеть

Архитектура свёрточной нейронной сети представляет собой упорядоченный набор слоёв, преобразующий представление изображения в другое представление. Основное отличие от обычных нейронных сетей состоит в том, что нейроны в одном слое будут связаны с небольшим количеством нейронов в предыдущем слое вместо того, чтобы быть связанными со всеми предыдущими нейронами слоя.

Первые 2 слоя – Conv2D. Эти сверточные слои будут работать с входными изображениями, которые рассматриваются как трёхмерные матрицы.
  32 и 64 это количество узлов в первом и втором слое.
  kernel — ядро свёртки, матрица весов. kernel_size соответственно размер данной матрицы(3х3)
  activation='relu' — функция активации для слоя. 
  Функция активации, которую мы будем использовать для первых двух слоев, называется ReLU (Rectified Linear Activation).
Flatten — слой выравнивания. Преобразует 2D данные в 1D данные. Он служит соединительным узлом между слоями.
Метод Dense соединяет слои между собой.
Метод Dropout помогает сети не переобучиться.


```python
x_train = trainingImages/255
y_train = keras.utils.to_categorical(trainingLabels, num_classes=2)
x_test = testImages/255
y_test = keras.utils.to_categorical(testLabels, num_classes=2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
```
Исходный код: [learnCNN.py](https://github.com/MAILabs-Edu-AI/lab-neural-networks-vision-dev-random/blob/master/learnCNN.py)

**Результаты**

```python
Train on 400 samples, validate on 200 samples

Epoch 20/20
400/400 [==============================] - 1s 2ms/step - loss: 0.0100 - accuracy: 0.9975 - val_loss: 0.0394 - val_accuracy: 0.9900
[0.03936159065924585, 0.9900000095367432]
```
Точность: 99%

*Необходимо также было сравнить результаты при использовании обучающего/тестового набора от разных людей (полученного в результате предыдущего задания), и при случайном разбиении всего набора данных в пропорции 80/20.*

Изначально тестовым набором был третий набор данных.

### Результаты для первого и второго набора в качестве тестовых: 

1) Полносвязная однослойная сеть:

Первый набор:

```python
Epoch 20/20
400/400 [==============================] - 0s 326us/step - loss: 0.0321 - accuracy: 0.9875
200/200 [==============================] - 0s 373us/step
[0.41026834726333616, 0.8899999856948853]
```
Точность: 88.99%

Второй набор:

```python
Epoch 20/20
400/400 [==============================] - 0s 242us/step - loss: 0.0310 - accuracy: 0.9900
200/200 [==============================] - 0s 569us/step
[0.11538490772247315, 0.9549999833106995]
```

Точность: 95.49%

| 1       | 2                     | 3       |
|---------|-----------------------|---------|
| 88.99% | 95.49% | 94.99% |


   2) Полносвязная многослойная сеть:
   
Первый набор:

```python
Epoch 20/20
400/400 [==============================] - 1s 2ms/step - loss: 0.0192 - accuracy: 0.9950
200/200 [==============================] - 0s 507us/step
[0.39357274293899536, 0.9049999713897705]
```
Точность: 90.04%

Второй набор: 

```python
Epoch 20/20
400/400 [==============================] - 0s 600us/step - loss: 0.0430 - accuracy: 0.9800
200/200 [==============================] - 0s 433us/step
[0.0975108927488327, 0.9599999785423279]
```
Точность: 95.99%

| 1       | 2                     | 3       |
|---------|-----------------------|---------|
| 90.04% | 95.99% | 95.49% |


   3) Свёрточная сеть:
   
Первый набор:

```python
Epoch 20/20
400/400 [==============================] - 1s 3ms/step - loss: 0.0067 - accuracy: 0.9975 - val_loss: 0.2833 - val_accuracy: 0.9200
[0.28334368077106775, 0.9200000166893005]
```
Точность: 92%

Второй набор: 

```python
Epoch 20/20
400/400 [==============================] - 1s 3ms/step - loss: 0.0074 - accuracy: 1.0000 - val_loss: 0.0303 - val_accuracy: 0.9900
[0.030285439938306808, 0.9900000095367432]
```
Точность: 99%

| 1       | 2                     | 3       |
|---------|-----------------------|---------|
| 92% | 99% | 99% |

### Случайное разбиение всего набора данных в пропорции 80/20:

Мы использовали метод shuffle для перемешивания набора данных и set/get_state для того, чтобы labels были связаны с самими изображениями и не перепутались при перемешивании. 

```python
rng_state = np.random.get_state()
np.random.shuffle(arr_images)
np.random.set_state(rng_state)
np.random.shuffle(arr_labels)

for i in range(0,479):
    trainingImages[i] = arr_images[i]
    trainingLabels[i] = arr_labels[i]

for i in range(480,599):
    testImages[i-480] = arr_images[i]
    testLabels[i-480] = arr_labels[i]
    ```

   1) Полносвязная однослойная сеть:

```python
Epoch 20/20
480/480 [==============================] - 0s 165us/step - loss: 0.0290 - accuracy: 0.9958
120/120 [==============================] - 0s 589us/step
[0.10340499877929688, 0.9666666388511658]

```
Точность: 96.66%(на 1.67% выше, чем выше, чем первоначальное разбиение) 

   2) Полносвязная многослойная сеть:
   
```python
Epoch 20/20
480/480 [==============================] - 0s 202us/step - loss: 0.0385 - accuracy: 0.9896
120/120 [==============================] - 0s 494us/step
[0.0776580199599266, 0.9833333492279053]
```
Точность: 98.33%(на 2.84% выше, чем первоначальное разбиение)

   3) Сверточная сеть:
   
```python 
Epoch 20/20
480/480 [==============================] - 1s 3ms/step - loss: 0.0132 - accuracy: 1.0000 - val_loss: 0.0250 - val_accuracy: 0.9917
[0.02499760507295529, 0.9916666746139526]
```
Точность: 99.16% > 99% (на 0.16% выше, чем первоначальное разбиение)

## Выводы

Данный проект помог нам освоить обработку изображений с помощью opencv и создание датасетов, а также обучение различных классификаций нейросетей.
Проделанная работа заставила задуматься над тем, что нейронные сети имеют огромные возможности.Правильно работающие НС могут значительно облегчить работу многих людей и организаций, так как их можно применять для решения поставленных задач во многих сферах.

Сложности, с которыми мы столкнулись:

1)при работе с изображениями при подготовке датасетов нам необходимо было исправить угол наклона символов с помощью специальных программ.

2)переобучение нашей нейронной сети, но после изменения вероятностей в методе dropout мы предотвратили взаимоадаптацию нейронов на этапе обучения, и наша нейронная сеть стала показывать хорошие результаты

Несмотря на то, что мы сейчас находимся на дистанционном обучении и не можем собраться все в одном месте, существуют различные программы для общей работы удаленно. Мы использовали беседу во ВКонтакте для обмена информацией для написания работы, конференцию в Skype для live-связи, а также Microsoft Visual Studio Live Share, чтобы каждый член команды мог писать и редактировать код онлайн.
