import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import gym
import numpy as np
import tensorflow as tf
import shutil
import logger
import json
from tensorflow import keras
from tensorflow.keras import layers
from atari_wrappers import make_atari, wrap_deepmind
from pathlib import Path

# Configuration parameters for the whole setup
discount_factor = 0.99  
max_steps_per_episode = 20

DEBUG = 10

env = make_atari("BreakoutNoFrameskip-v4")
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
# preprocesses sequence, warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
seed = 2
env.seed(seed)

num_actions = env.action_space.n 
print("Numb of actions " + str(num_actions))

def create_A2C_model():
    inputs = layers.Input(shape=(84, 84, 4,))
    # Convolutions on the frames on the screen
    model = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    model = layers.Conv2D(64, 4, strides=2, activation="relu")(model)
    model = layers.Conv2D(64, 3, strides=1, activation="relu")(model)

    model = layers.Flatten()(model)

    model = layers.Dense(512, activation="relu")(model)

    action = layers.Dense(num_actions, activation="softmax")(model)
    critic = layers.Dense(1, activation="linear")(model)

    return keras.Model(inputs=inputs, outputs=[action, critic])

A2C_model = create_A2C_model()

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
behaviour_running_reward = 0
episode_count = 0

# Setup logging for the model
logger.set_level(DEBUG)
dir = "logs"
if os.path.exists(dir):
    shutil.rmtree(dir)
logger.configure(dir=dir)

# replay memory
state_history = []
exp_reward_history = []
done_history = []

episodes_history = []
episodes_history_max_length = 1000

min_episodes_memory = 1

while True:  # Run until solved
    state = np.array(env.reset())
    behaviour_episode_reward = 0
    # Behaviour policy and storing of experiences to experience replay
    for timestep in range(1, max_steps_per_episode):
        #env.render()
        # Behaviour Policy, follow current policy
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probs, critic_value = A2C_model(state)

        action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        state, reward, done, _ = env.step(action)
        
        # Save experience to experience replay buffer
        state_history.append(state)
        exp_reward_history.append(reward)
        done_history.append(done)
        behaviour_episode_reward += reward
        if(timestep == 1000):
            break
    
    behaviour_running_reward = 0.05 * behaviour_episode_reward + (1 - 0.05) * behaviour_running_reward 

    # Adds episode data to replay buffer
    episodes_history.append([state_history, exp_reward_history, done_history])
    # Resets behaviour policy state for the episode
    state_history = []
    exp_reward_history = []
    done_history = []
    # Delete latest oldest episode if buffer too lage    
    if len(episodes_history) > episodes_history_max_length:
        del episodes_history[:1]

    
    replay_length = len(episodes_history)

    if(replay_length > min_episodes_memory):
        indice = np.random.choice(range(replay_length), size=1)[0]
        # episodes_history[indice][0] indices: episode history, states, number of states for that episode
        replay_episode_length = len(episodes_history[indice][0])
        with tf.GradientTape() as tape:
            episode_reward = 0
            for experience_time_step in range(replay_episode_length):
                # still needs to only update gradient every 4th frame for effeciency
                # Predict action probabilities and estimated future rewards
                # from environment state
                state_index = 0
                reward_index = 1

                state = episodes_history[indice][state_index][experience_time_step]
                reward_exp = episodes_history[indice][reward_index][experience_time_step]

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                
                action_probs, critic_value = A2C_model(state)

                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution
                # squeeze can transform tensor to np array
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                # Appends/ saves natural log of his action probability, ex ln(0.26182014) = -1.3466
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                # Apply the sampled action in our environment
                rewards_history.append(reward_exp)
                episode_reward += reward_exp

            # Update running reward to check condition for solving
            # 5% weight from newest episode + 95% of reward from older episodes, when "average" running reward is high enough, stop 
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward 

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with discount_factor
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + discount_factor * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                # advantage, with discounted return we got from one action, and the value our critic expected from that state 
                advantage = ret - value 
                actor_losses.append(-log_prob * advantage)  # actor loss according to loss formular https://imgur.com/a/G7zbUtV

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards. Compares its value prediction vs the actual recieved discounted reward.
                # Critic loss function https://imgur.com/a/pRc5VLF.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # ----------------------------------------------- #
            # LOSS FOR BOTH COMBINED
            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)

            grads = tape.gradient(loss_value, A2C_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, A2C_model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()
    
    if episode_count % 1 == 0:
        # Log every episode 
        logger.logkv("behaviour_running_reward", behaviour_running_reward)
        logger.logkv("replay_running_reward", running_reward)
        logger.logkv("episode", episode_count)
        logger.dumpkvs()

    # Put it here to avoid saving on 0th episode lol
    episode_count += 1

    # Save Model every 100th episode
    if(episode_count % 100 == 0):
        model_path = 'models/A2C-episode-{}/'.format(episode_count)
        # Path converting if working on windows
        # model_path = Path("source_data/text_files/") <--- Creates path that works on both windows and unix
        # model_path = str(Path(model_path))
        print("Saved model at episode {}".format(episode_count))
        # Save tensorflow model
        A2C_model.save(model_path)

        # Save the parameters
        data = { "running_reward": running_reward, "episode" : episode_count}
        data = json.dumps(data)    
        param_file = json.loads(data)

        actor_file ='{}\data.json'.format(model_path)
        with open(actor_file,'w+') as file:
            json.dump(param_file, file, indent = 4)
