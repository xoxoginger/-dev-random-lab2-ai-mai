import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import keras.layers.advanced_activations
import numpy as np
import pathlib
import cv2

trainingImages = np.empty([400, 28,28,3])
trainingLabels = np.empty([400])

testImages = np.empty([200, 28,28,3])
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
  arr = np.asarray(image)
  trainingImages[i-100] = arr
for i in range(300, 499):
  image = cv2.imread("cropping/dataset/data2/" + str(i) +".png")
  arr = np.asarray(image)
  trainingImages[i-100] = arr
for i in range(500, 699):
  image = cv2.imread("cropping/dataset/data3/" + str(i) +".png")
  arr = np.asarray(image)
  testImages[i-500] = arr

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
print(score)