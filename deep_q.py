from collections import deque #used for replay memory
import tensorflow as tf
import numpy as np  # Python ≥3.5 is required
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from tensorflow import keras
import tensorflow as tf
import sklearn
import sys
import gym # access to ai_gym library
assert sys.version_info >= (3, 5)


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# TensorFlow ≥2.0 is required
assert tf.__version__ >= "2.0"


# to make this files output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# to plot pretty pictures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
mpl.rc('animation', html='jshtml')


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

# Good practice to clear any previous tf session
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)


# You need a neural net that takes a state-action pair
# and outputs an approximate Q-Value. 
# For this example, we use a neural net that takes a 
# state and outputs one approximate Q-Value for each possible action.
env = gym.make("CartPole-v1")
input_shape = [4]  # == env.observation_space.shape
n_outputs = 2  # == env.action_space.n

# For this environment we use a simple neural net, with two hidden layers
model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

# To select an action using this DQN, we pick the action with the 
# largest predicted Q-Value. We use the e-greedy policy to ensure
# that the agent explores the environment.
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


# we store all of the training experiences of the DQN and 
# store them in a replay buffer. We will sample a random 
# training bactch from it at each training iteration. 
# This helps reduce the correlations between experiences in 
# a batch. A deque list is used for the process
replay_memory = deque(maxlen=2000)


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error


def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rewards = []
best_score = 0

for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step)  # Not shown in the book
    if step > best_score:  # Not shown
        best_weights = model.get_weights()  # Not shown
        best_score = step  # Not shown
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(
        episode, step + 1, epsilon), end="")  # Not shown
    if episode > 50:
        training_step(batch_size)

model.set_weights(best_weights)

plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
save_fig("dqn_rewards_plot")
plt.show()


env.seed(42)
state = env.reset()

frames = []

for step in range(200):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)

plot_animation(frames)
