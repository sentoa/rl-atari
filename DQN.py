import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import gym
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import logger
import os
import shutil
import json
import argparse
import math

#print(tf.executing_eagerly())
#tf.config.run_functions_eagerly(True)
tf.autograph.set_verbosity(0)

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 0.9  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

DEBUG = 10

"""# Q Model"""

def create_q_model(num_actions):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    model = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    model = layers.Conv2D(64, 4, strides=2, activation="relu")(model)
    model = layers.Conv2D(64, 3, strides=1, activation="relu")(model)

    model = layers.Flatten()(model)

    model = layers.Dense(512, activation="relu")(model)
    action = layers.Dense(num_actions, activation="linear")(model)

    return keras.Model(inputs=inputs, outputs=action)

"""# Generator Model """
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(21*21*128, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((21, 21, 128)))
    assert model.output_shape == (None, 21, 21, 128)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 42, 42, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(4, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 84, 84, 4)

    return model

"""# Descriminator Model """
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[84, 84, 4]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def random_generate_batch():
    model = make_generator_model()

    noise = tf.random.normal([1, 100])
    generated_state = model(noise, training=False)
    return tf.squeeze(generated_state)
    """ #FOR VISUALIZATION
    state = np.array(generated_image)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = np.random.random(size=(3, 3, 3))
    z, x, y = state.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()"""
"""# Setup Environment and Instanciate Q Models"""

# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)


# Initialize Q-network and target network with random weights
# The Q-network makes the predictions which are used to take an action.
model = create_q_model(env.action_space.n)
model_target = create_q_model(env.action_space.n)


# Initialize Descriminator and Generator Models
generator_model = make_generator_model()
discriminator = make_discriminator_model()


"""# Configuration"""

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def discriminator_loss(real_output, fake_output, reward_list):
    # IMAGE LOSS
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    image_loss = real_loss + fake_loss
    #image_loss = tf.math.log(image_loss)

    # REWARD LOSS
    reward_avg = np.average(reward_list)

    if reward_avg == 0:
        reward_loss = tf.constant(1, dtype=tf.float32)
    else:
        reward_loss = tf.constant(reward_avg, dtype=tf.float32)
        reward_loss = tf.math.log(reward_loss)

    # TOTAL LOSS
    total_loss = image_loss - reward_loss
    
    return image_loss

@tf.function
def train_step(real_image, reward_list):
    noise = tf.random.normal([1, 100])

    with tf.GradientTape() as disc_tape:
      
        generated_images = generator_model(noise, training=False)

        real_output = discriminator(real_image, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output, reward_list)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Number of episodes
episodes = 100000

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# Number of frames to take random action and observe output

epsilon_frame_cap = 1000

# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000

# Train the model after 4 actions
update_after_actions = 4

# How often to update the target network
update_target_network = 1000

# Epsilon Greedy Factor - Lower number means more random actions will be taken
epsilon_factor = 10000000

# Using huber loss for stability
loss_function = keras.losses.Huber()

# Setup logging for the model
logger.set_level(DEBUG)
dir = "logs"
if os.path.exists(dir):
    shutil.rmtree(dir)
logger.configure(dir=dir)

for episode in range(episodes):
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        #env.render(); 
        frame_count += 1

        # Use epsilon-greedy for exploration
        # In the first X frames, epsilon_frame_cap will always take random action,
        # since we do not have any Q values yet.
        # Otherwise, we use the random distribution from epsilon-greedy.
        if frame_count < epsilon_frame_cap or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(env.action_space.n)
        else:
            # Predict action Q-values From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon / epsilon_factor
        # If enough timesteps have been reached, we do not want epsilon
        # to reach 0, so we ensure there is a minimum threshold
        epsilon = max(epsilon, epsilon_min)


        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        replay_length = len(action_history)

        # Update every fourth frame and we need to fill out the dataset with
        # enough data. For this, we just sample batch
        if frame_count % update_after_actions == 0 and replay_length > batch_size:

            # Get indices of samples for replay buffers
            # Take randomly batch size amount of states
            indices = np.random.choice(range(replay_length), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            for i in indices:
                real_image = tf.convert_to_tensor(state_history[i], dtype=tf.float32)
                real_image = tf.expand_dims(real_image, axis=0)
                train_step(real_image, rewards_sample)

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            # We insert 32 batch states into the neural model, outputs all
            # 32 states with each of
            future_rewards = model_target.predict(state_next_sample)


            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            updated_q_values = updated_q_values * (1 - done_sample) - done_sample


            # Create a mask so we only calculate loss on the updated Q-values
            # Translate each action to a 1 matrix. So for action 2 it is 
            # [0. 1. 0. 0.]. Action 3 [0. 0. 1. 0.] In batch size length
            masks = tf.one_hot(action_sample, env.action_space.n)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)


            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # Log every episode 
    logger.logkv("reward", running_reward)
    logger.logkv("episode", episode_count)
    logger.logkv("frame_count", frame_count)
    logger.dumpkvs()

    episode_count += 1

    # Save Model every 100th episode
    if(episode_count % 10 == 0):
        print("Saved model at episode {}".format(episode_count))
        model_path = 'models/episode-{}'.format(episode_count)

        # Save tensorflow model
        model.save(model_path)

        # Save Descriminator Model
        checkpoint_dir = 'checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(discriminator_optimizer=discriminator_optimizer,
                                        discriminator=discriminator)
        checkpoint.save(file_prefix = checkpoint_prefix)

        # Save the parameters
        data = { "running_reward": running_reward, "episode" : episode_count,
                 "frame_count" : frame_count}
        data = json.dumps(data)    
        param_file = json.loads(data)
        filename='{}/data.json'.format(model_path)
        with open(filename,'w+') as file:
            json.dump(param_file, file, indent = 4)

