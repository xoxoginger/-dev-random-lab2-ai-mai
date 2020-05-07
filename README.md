# Нейронные сети: лабораторная работа

Лабораторная работа выполняется в группах по 3 человека. 

**Цель работы:** подготовить набор данных и построить несколько нейросетевых классификаторов для распознавания рукописных символов. 

Варианты рукописных символов (номер определяется следущим образом: ASCII-коды первых символов фамилий всех участников команды складываются и берётся остаток от деления на 5 + 1):

1. Символы принадлежности множеству, пересечения, объединения множеств и пустого множества.
2. 5 первых символов греческого алфавита
3. Улыюбающиеся, хмурящиеся и нейтральные смайлики
4. Значки, использующиеся при голосовании: checkmark и крестик
5. Буквы Э, Ю, Я.

Работа состоит из следующих заданий:

### Подготовка данных

1. На листе бумаги в крупную клеточку каждый участник команды пишет по 100 символов каждого класса.
2. Лист очень качественно фотографируется (без искажений, при необходимости с использованием программ типа Office Lens), изображение при необходимости обрабатывается в редакторе (повышение контрастности)
3. С помощью Python-скрипта из изображения вырезаются отдельные символы в виде изображений размерности 32\*32. Обратите внимание, что библиотека OpenCV позволяет загрузить изображение как `numpy`-массив размерности M\*N\*3, после чего вырезание работает простым взятием Python-среза от такого массива (с фиксированным шагом)

В результате должен получится набор фотографий, примерно по 300 фотографий для каждого класса. Набор данных надо разбить на обучающий и тестировочный набор, включив в обучающий набор символы, написанные 2 участниками команды, а в тестовый -- символы, написанные третьим участником.

Рекомендуемая ссылка: [пример работы с изображениями в OpenCV](https://arboook.com/kompyuternoe-zrenie/osnovnye-operatsii-s-izobrazheniyami-v-opencv-3-python/)

### Обучение полносвязной нейронной сети 

На основе полученного набора данных обучить полносвязную нейросеть с одним слоем и с несколькими слоями. Рекомендуется использовать фреймворк [Keras](https://keras.io/), как наиболее простой. Примеры можно найти в репозитории http://github.com/shwars/NeuroWorkshop, или [в этой статье](https://habr.com/ru/company/wunderfund/blog/314242/)

По результатам обучения необхолимо сравнить результаты, попробовать подобрать параметры (количество слоёв, нейронов в каждом слое, параметры алгоритма обучения) и проанализировать результаты.

Необходимо также сравнить результаты при использовании обучающего/тестового набора от разных людей (полученного в результате предыдущего задания), и при случайном разбиении всего набора данных в пропорции 80/20.

### Обучение свёрточной нейронной сети

Обучите свёрточную нейронную сеть для решения такой же задачи классификации. Сравните результаты. Пример свёрточной нейронной сети на Keras можно найти [в документации](https://keras.io/examples/mnist_cnn/).

### Написание отчёта

В результате работы необходимо написать отчёт (в файле Readme.md), а также приложить исходные фотографии, полученный набор данных и написанный код в виде python-файлов или Notebooks. В отчёте не забудьте отразить вклад каждого участника команды в результат, а также выводы.
