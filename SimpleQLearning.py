#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:20:35 2020

@author: abhinand
"""

import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()
LR = 0.1
DISCOUNT = 0.95
EPISODES = 8000
SHOW_EVERY = 2000
done = False
EPSILON = 0.5
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES//2
EPSILON_DECAY_VALUE = EPSILON/(END_EPSILON_DECAY - START_EPSILON_DECAY)

#print(env.observation_space.high) #Printing upper and lower bounds for states
#print(env.observation_space.low)
#print(env.action_space.n)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) #Converting continuous values to [20,20] discrete values for each obse
discrete_os_size_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE #Getting the window size of distance and velocity

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n])) #Populating the q_table with values [-2,0)
#print(q_table.shape)

def get_discrete_state(state): #function to get discrete values of states
    discrete_state  = (state - env.observation_space.low)/discrete_os_size_win_size
    return tuple(discrete_state.astype(np.int))
#print(q_table[get_discrete_state(env.reset())]) #Access Q table value for given state

for epoch in range(1,EPISODES+1):
    discrete_state = get_discrete_state(env.reset())  
    done = False
    if epoch%SHOW_EVERY == 0:
        print(epoch)
        render = True
    else:
        render = False
    while not done:
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[discrete_state]) #Action with max Q -> New action
        else:
            action = np.random.randint(0,env.action_space.n)
        if render == True:
            env.render()
        new_state,reward,done,_ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            curr_q = q_table[discrete_state + (action,)]
            
            new_q = (1 - LR)*curr_q + LR*(reward + DISCOUNT*max_future_q) #Q-Learning equation
            
            q_table[discrete_state + (action,)] = new_q #Update Q value
        elif new_state[0] >= env.goal_position: #Checking if we reached the hilltop
            q_table[discrete_state + (action,)] = 0 #Setting Q value to the max value ie 0
        discrete_state = new_discrete_state
    if END_EPSILON_DECAY >= epoch >= START_EPSILON_DECAY:
        EPSILON -= EPSILON_DECAY_VALUE
  

env.close()