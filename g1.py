import gym
import numpy as np
import random

def action(ob, reward, done):
    """
    0   Cart Position   -2.4    2.4
    1   Cart Velocity   -Inf    Inf
    2   Pole Angle  ~ -41.8°    ~ 41.8°
    3   Pole Velocity At Tip
    """
    if (ob[2] > 0 and ob[3] > -1) or ob[3] > 1:
        return 1
    return 0


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    episode_count = 1
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            ob, reward, done, _ = env.step(action(ob, reward, done))
            env.render()
            print(ob, reward, done)
