# to_categorical 이해하기

from keras import models
from keras import layers
import numpy as np
from keras.utils import to_categorical

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

y = to_categorical(y) # tensorflow version==2.0.0에서는 y = tf.keras.utils.to_categorical(y)

print(y)