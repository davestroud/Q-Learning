import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam


""" 
Q-Learning can only work if the exploration policy explores
the MDP thoroughly enough. We use the e-greedy policy, e is 
epsilon:at each step to act randomly with probability. 

The advantage of the e-greedy policy is that is will spend more
and more time exploring interesting parts of the environment, 
as the Q-Value estimates get better and better, while still 
spending some time visiting unknown regions of the MDP. It is
common to start with a high value for e (e.g.,1.0) and then
gradually reduce it down to 0.05.
"""
epsilon = .1 # max(1-episode / 500, 0.01)
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

def guess_number(model, guess_vec):
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, num_targets, size=1)[0]
    else:
        prob = model.predict(np.array([guess_vec]))[0]
        print(prob)
        action = np.random.choice(num_targets, 1, p=prob)[0]
    return action

def generate_batch(history):
    M,L = [],[]
    for i, idx in enumerate(np.random.randint(0, len(history),
                            size = min(len(history), batch_size))):
        cc, a, r, _, _ = history[idx]
        M.append(cc)
        L.append(model.predict(np.array([cc]))[0])
        L[i][a] = r
    return np.array(M), np.array(L)
    