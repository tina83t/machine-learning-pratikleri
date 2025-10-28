import gym 
import numpy as np 
import random

from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode = "ansi")
env.reset()
print(env.render())

"""
0: west
1: north
2: east
3: south
4: pick up 
5: drop off
"""

action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space,action_space))

alpha = 0.1 #learning rate
gamma = 0.6 #discount rate
epsilon = 0.1 #epsilon

for i in tqdm(range(1,1000)):
    state , _ = env.reset()
    
    done = False
    
    while not done:
        if random.uniform(0,1) < epsilon :  #explore
            pass
        else : #exploit
            action = np.argmax(q_table[state])
            
        next_state , reward , done , info , _ = env.step(action)
            
        q_table[state, action] = q_table[state,action]+ alpha *(reward+gamma * np.max(q_table[next_state]) - q_table[state,action])    

        state = next_state
        
print("training finished")        

# test
total_epoch, total_penalties = 0, 0
episodes = 100

for i in tqdm(range(episodes)):
    state , _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
       
        next_state , reward , done , info , _ = env.step(action)
            
        state = next_state
        
        if reward == -10:
            penalties += 1
            
            
        epochs += 1
        
    total_epoch += epochs
    total_penalties += penalties
    
print("result after {} episodes".format(episodes))
print("avrage timesteps per episode :",total_epoch/episodes)
print("avrage penalties per episode :",total_penalties/episodes)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        