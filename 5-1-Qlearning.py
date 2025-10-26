import gym 
import random
import numpy as np

environment = gym.make ("FrozenLake-v1", is_slippery = False , render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions= environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)

action = environment.action_space.sample()

"""
left = 0
down = 1
right = 2
up = 3
"""
# S1 -> (Action 1)-> S2

new_state, reward, done, info, _ = environment.step(action)

#%%

import gym 
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

environment = gym.make ("FrozenLake-v1", is_slippery = False , render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions= environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)

episodes = 1000 
alpha = 0.5 
gamma = 0.9

outcomes = []

#training 
for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False 
    
    outcomes.append("failure")
    
    while not done :
        if np.max(qtable[state])> 0:
            action = np.argmax(qtable[state])
        else :
                action = environment.action_space.sample()
        new_state, reward, done, info, _ = environment.step(action) 
        
        # q table update
        qtable[state, action] = qtable[state, action]+alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state,action])
        
        state = new_state
        
        if reward:
            outcomes[-1] = "success"
            
print("QQtable After Training: ")
print(qtable)

plt.bar(range(episodes),outcomes)




