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
        Q_values = model.predict(state[np.newaxis]) # used to increase dim of existing array by one more dim
        return np.argmax(Q_values[0]) # Returns indices of max element of array in a particular axis


# we store all of the training experiences of the DQN and 
# store them in a replay buffer. We will sample a random 
# training batch from it at each training iteration. 
# This helps reduce the correlations between experiences in 
# a batch. A deque list is used for the process
replay_memory = deque(maxlen=2000)


"""
    Each experience will be composed of five elements
        - a state
        - the action the agent took
        - the resulting reward
        - the next state it reached
        - a Boolean indicating whether the episode ended at that point (done).
"""

# Sample a random batch of experiences from the replay buffer. 
# This fuction will return five NumPy arrays corresponding to 
# the five experience elements. 
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

# Create a function that will play a single step using the e-greedy policy. 
# Store the resulting experience in the replay buffer
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

# Define hyperparameters
# Create optimizer and loss function
batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

"""
    training_step

    starts by sampling a batch of experiences, then uses the DQN
    to predict the Q-Value for each possible action in each experiences
    next state. 
    Assuming the agent with play optimally, we only keep the maximum Q-Value 
    for each next state.
    Next, use the Target Q-Value function to compute the target Q-Value
    for each experiences state-action pair. 

    Next, use the DQN to compute the Q-Value for each experienced state-action pair.
    This results in the DQN outputting the Q-Values for the other possible actions, 
    not just for the action that was chosen by the agent. Thus, we need to mask out 
    all of the Q-Values that we don't need. The tf.one_hot() function makes it easy 
    to convert an array of action indices into such a mask.

    Next, we compute the lost: it is the mean squared error between the target and 
    predicted Q-Values for the experienced state-action pairs.

    Finally, we perform a Gradient Descent step to minimize the loss with regard to 
    the model's trainable variables.

"""
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



"""
     Train the model
        We run 600 episodes, each with a maximum of 200 steps. At each step, we 
        first compute the epsilon value for the e greedy policy: it goes from 1
        down to 0.01, linearly, in a bit under 500 episodes

        Next, call the play_one_step function, which will use the e-greedy policy 
        to pick an action, then execute it and record the experience in the replay buffer.

        If the episode is done, we exit the loop.

        Finally, if we are past the 50th episode, we call the training_step() function
        to train the model on one branch sample from the replay buffer. 
 """


for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step)  
    if step > best_score: 
        best_weights = model.get_weights()  
        best_score = step  
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(
        episode, step + 1, epsilon), end="")  
    if episode > 50: # Gives the replay buffer time to fill up
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
