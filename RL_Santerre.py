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

num_targets = 10
history = []

model = Sequential()
model.add(Dense(hidden_size, input_shape=(len(target),), activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(num_targets, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))
