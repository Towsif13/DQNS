from agent import Agent
import gym
import torch
import random
from environment import Environment
import time

agent = Agent(state_size=4, action_size=4, seed=0)
# env = gym.make('LunarLander-v2')
# env.seed(0)
env = Environment()
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(10):
    state = env.reset()
    total_reward = 0
    for j in range(200):
        action = agent.act(state)

        state, reward, done = env.step(action)
        env.render()
        time.sleep(0.4)
        # print(
        #     f"reward : {reward} Drone position: {state} Man position: {env.man_x, env.man_y}")
        total_reward += reward
        print(
            f"reward : {reward} Total rewaard : {total_reward:.2f}")
        if done:
            time.sleep(1)
            break

# for j in range(15):
#     for i in range(5):
#         state = env.reset()
#         for j in range(200):
#             action = agent.act(state)
#             env.render()
#             state, reward, done, _ = env.step(action)
#             if done:
#                 break
