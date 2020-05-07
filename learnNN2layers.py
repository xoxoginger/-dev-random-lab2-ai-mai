import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.layers.advanced_activations
import numpy as np
import pathlib
import cv2

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
print(score)