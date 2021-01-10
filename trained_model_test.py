#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:32:07 2020

@author: abhinand
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:50:00 2020

@author: abhinand
"""

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import gym

DISCOUNT = 0.95
MAX_REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'MountainCar'
MIN_REWARD = -50 # For model save
#MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 2000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True


'''
This is our own environment on which we will work!
'''




env = gym.make('MountainCar-v0')

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


'''
We're doing this to keep our log writing under control. Normally, Keras 
wants to write a new logfile per .fit() which will give us a new ~200kb file 
per second. That's a lot of files and a lot of IO, where that IO can take 
longer even than the .fit(). So this class will help in maintaining a single
file for all the fits.
'''

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class DQNAgent:
    def __init__(self):
        #main model, this is what we train every step
        self.model = self.create_model()
        #Target model, this is what we .predict every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen = MAX_REPLAY_MEMORY_SIZE) #To create a batch of input states
        
        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")
        
        self.target_update_counter = 0 #Counter for updating target model internally
        
    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(env.observation_space.shape[0],), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(env.action_space.n, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics = ['accuracy'])
                
        return self.model
    
    def update_replay_memory(self, transition): #Function to add observation states into the replay_memory queue
        self.replay_memory.append(transition) #(observation space, action, reward, new observation space, done)
        
    def get_qs(self,state): #To get Q values
        return self.model.predict(np.reshape(state, [1,env.observation_space.shape[0]]))[0]
    
    def train(self, terminal_state, step): #Function to train our model
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) #Minibatch is a random sample of replay_mem
        
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states) #Needed for Q calc (Q learning eqn.)
        
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states) #Needed for Q calc (Q learning eqn.)
        
        X = [] #Inputs - pixels
        Y = [] #Labels - actions
        
        for index, (current_state, action, reward, new_current_state,done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list)
                new_q = reward + DISCOUNT * max_future_q
            else: #If done then new q will be the reward
                new_q = reward 
                
            current_qs = current_qs_list[index]
            current_qs[action] = new_q #Updating the previous Q value with the new calculated value
            
            X.append(current_state)
            Y.append(current_qs)
        #Fit only it terminal_state(Done) = None
        self.model.fit(np.array(X),np.array(Y),batch_size = MINIBATCH_SIZE, verbose = 0, 
                       shuffle = False, callbacks = [self.tensorboard] if terminal_state else None)
        
        
        if terminal_state:
            self.target_update_counter += 1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
        
agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0,env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        
        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        
        
        if SHOW_PREVIEW and episode>1500 and not episode % AGGREGATE_STATS_EVERY:
            env.render()
            
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        if episode > 1000 and not episode % AGGREGATE_STATS_EVERY:
            print(max_reward)
        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    
    if episode == 2000:
        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
