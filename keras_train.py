# 문제 3
import keras 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

y = np_utils.to_categorical(y)

model = models.Sequential()

model.add(layers.Dense(256, input_dim = 1, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x,y,
                    epochs=50,
                    batch_size=1)

x_prd = np.array([11, 12, 13])
print(model.predict(x_prd))