import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam,sgd
epsilon = .1
epochs = 1000
hidden_size = 10
batch_size = 50
max_hist = 250
discount = .5
num_targets = 6
target = np.array([1,2])
target_len = len(target)
history = []
model = Sequential()
model.add(Dense(hidden_size, input_shape=(target_len,), activation='elu'))
model.add(Dense(hidden_size, activation='elu'))
model.add(Dense(num_targets))
model.compile(sgd(lr=.2), "mse")
def guess_number(model, guess_vec):
  if np.random.rand() <= epsilon:
    action = np.random.randint(0, num_targets, size=1)[0]
  else:
    action = np.argmax(model.predict(np.array([guess_vec]))[0])    
  return action
def generate_batch(history):
  M,L=[],[]
  for i, idx in enumerate(np.random.randint(0, len(history),
                          size=min(len(history), batch_size))):
      state, action, reward, next_state, game_over = history[idx]
      M.append(state)
      L.append(model.predict(np.array([state]))[0])
      Q_sa = np.max(model.predict(next_state.reshape(1,-1)))
      if game_over:  
        L[-1][action] = reward
      else:
        L[-1][action] = reward + (discount * Q_sa)
  return np.array(M), np.array(L)
for epoch in range(epochs):
  game_over = False
  current_game = np.zeros(target_len)
  for x in range(target_len):
    guess = guess_number(model, current_game)
    tmp_game =  current_game.copy()
    current_game[x] = guess
    if x == target_len-1: # if last judgement
      game_over = True
    if game_over:
      print(current_game)
      if not np.array_equal(current_game,target):
        reward = -1 
      else:
        reward = 1
    else:
        reward = 0
    history.append([tmp_game, guess, reward, current_game.copy(), game_over])
    M, L = generate_batch(history)
    model.train_on_batch(M, L)
    if len(history)>max_hist:
      del history[0]
import pdb; pdb.set_trace()