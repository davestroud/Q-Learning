import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam


epsilon = .1
epochs = 10000
hidden_size = 50
batch_size = 100
max_hist = 500


target = np.array([4])
