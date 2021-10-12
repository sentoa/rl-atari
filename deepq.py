import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

start_time = time.time()

#max_memory_length = 150
max_memory_length = 50000 # should be 1000000 to be similar to deepmind, check if can run without memory issues
batch_size = 32
gamma = 0.99
frame_count = 0
episode_count = 0
# only take a new action after the same action have been repeated for 4 frames
action_repeat = 4
max_steps_per_episode = 10000
update_target_network = 10000
loss_function = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
# rms prop optimizer
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00025)
# Definies epsilon
epsilon_greedy_frames = 1000000.0
epsilon_random_frames = 50000 
#epsilon_random_frames = 0
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken

#epsilon = 0
recent_episodes_mean_reward = 0


# update_target_network = 100

debug_episodes = 7

# Initilazises breakout as environment
'''v0 has a repeat action probability of 0.25, meaning that a quarter of the time the previous action will be used instead of the chosen action. These so-called sticky actions are a way of introducing stochasticity into otherwise deterministic environments, largely for benchmarking.
v4 always performs the chosen action.'''
'''No suffix: skip 2, 3 or 4 frames at each step, chosen at random each time
Deterministic: skip 4 frames at each step
NoFrameskip: skip 0 frames at each step'''
env = make_atari("BreakoutNoFrameskip-v4")

# preprocesses sequence, warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
seed = 2
env.seed(seed)

num_actions = len(env.unwrapped.get_action_meanings())
print("Numb of actions " + str(num_actions))
# prifire bt(env.unwrapped.get_action_meanings())
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT']


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)



# Initialize action-value function with random weights
#model = keras.models.load_model('E:\\GithubEDrive\\sw9\\models')
model = create_q_model()
# Initialize target action-value function with random weights
#model_target = keras.models.load_model('E:\\GithubEDrive\\sw9\\models')
model_target = create_q_model()

# Initialize Replay Memory
action_history = []
state_history = []
next_state_history = []
reward_history = []
done_history = []
episode_reward_history = []

while True:
    state = np.array(env.reset()) # Reset returns an initial observation, in breakout, resets ball and agent
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        #env.render(); 
        #time.sleep(.01)
        frame_count += 1
        
        # With probability epsilon select random action
        if(frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]):
            action = np.random.choice(num_actions)
        
        # Otherwise select argmax of deep q function
        else:
            state_tensor = tf.convert_to_tensor(state) # Converts numpy array that represents environment to tf tensor #(84, 84, 4)
            state_tensor = tf.expand_dims(state_tensor, 0) # (1, 84, 84, 4) interesting it needs 1 to fit the input of the model
            action_probs = model(state_tensor, training=False)
            
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Execute action in emulator and observe reward and next state 
        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state) # WHY IS THIS NECESSARY
        
        episode_reward += reward
        
        # Store transition in replay memory
        action_history.append(action)
        state_history.append(state)
        next_state_history.append(next_state)
        reward_history.append(reward)
        done_history.append(done)
        state = next_state

        # Deletes oldest entry in replay memory buffer if buffer is full
        if(len(action_history) > max_memory_length):
            del action_history[:1]
            del state_history[:1]
            del next_state_history[:1]
            del reward_history[:1]
            del done_history[:1]

        # start random sampling minibatches if memory replay buffer is larger than a batch
        if(frame_count % action_repeat == 0 and len(action_history) > batch_size):
            # get random indices for sampling
            # first argument interval to get random numbers from
            # second argument numpy array size 
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Since indice array is static, first sample will be index 0 for sample arrays, second sample index 1 and so on
            # states need to be represented by np.arrays since they have shape (32, 84, 84, 4), actions and rewards are just a number.
            state_sample = np.array([state_history[i] for i in indices])
            next_state_sample = np.array([next_state_history[i] for i in indices])
            
            action_sample = [action_history[i] for i in indices]
            rewards_sample = [reward_history[i] for i in indices] 

            # Because we multiply done sample with our model prediction which is a tensor later
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Use frozen model to predict expected future reward which will be used to compute target  
            future_rewards = model_target.predict(next_state_sample)

            # Computes Q for target = reward + discount factor * expected future reward
            # print(len(rewards_sample)) # 32, every timestep in the episode it gets a batch of 32 rewards that it uses to perform gradient descent and update the model
            # print(future_rewards.shape) (32, 4) after transformation (32,) where it have chosen the best action of each action space
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            #print(type(updated_q_values))
            # If final frame set the last value to -1 - if atari breakout lost ball in previous frame - q value is a negative reward
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                #print(state_sample.shape)
                q_values = model(state_sample)
                #print("q values before mask")
                #print(q_values)
                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                #print("multiplied q values by mask")
                #print(tf.multiply(q_values, masks))
                
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)
                #print(loss)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"

            print(template.format(recent_episodes_mean_reward, episode_count, frame_count))
        
        if done:
            break
    
    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    recent_episodes_mean_reward = np.mean(episode_reward_history)
    
    episode_count += 1  

    #print("Episode {} reward: {}".format(episode_count + 1, episode_reward))
    #print("Replay buffer size: " + str(len(action_history)))
    
    if(episode_count == 50):
        time_taken = time.time() - start_time
        print("time taken: " + str(time_taken))

    if(episode_count % 1000 == 0):
        print("saved model at episode " + str(episode_count)) 
        model.save('models')
        
    if recent_episodes_mean_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        #model.save('E:\\GithubEDrive\\sw9\\models')
        break
  
    
    # or episode count reached - for debuggimg purposes
    #if(episode_count == debug_episodes):
    #    break


