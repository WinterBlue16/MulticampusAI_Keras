import numpy as np
import keras

x = np.array([range(1, 51),range(51, 101), range(101, 151), range(151, 201)])
y = np.array([1, 51, 101, 151])

print(x.shape)
print(y.shape)