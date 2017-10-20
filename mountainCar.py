import gym
import os 
import random
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam 
from collections import deque

weight_backup = 'weights/mountainCar.h5'

class Agent():
    def __init__(self, action_size, state_size):
        self.momery = deque(maxlen=2000)
        self.action_size = action_size
        self.state_size = state_size
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.999
        self.sample_bath_size = 60
        self.brain = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(weight_backup):
            model.load_weights(weight_backup)
            self.exploration_rate = self.exploration_min
        return model 


    def save_model(self):
        self.brain.save(weight_backup)


    def act(self, state):
        if np.random.rand() <= self.exploration_rate: 
            return random.randrange(self.action_size)

        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])


    def remember(self, state, action, reward, next_state, done, score):
        self.momery.append((state, action, reward, next_state, done, score))


    def rethink(self, is_fit=True):
        if len(self.momery) < self.sample_bath_size:
            return

        sample_bath = random.sample(self.momery, self.sample_bath_size)
        for state, action, reward, next_state, done, score in sample_bath:
            target = score
            if not done:
                target = score + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target 
            if is_fit:
                self.brain.fit(state, target_f, epochs=1, verbose=0)
        print(target)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay



class MountainCar(object):
    def __init__(self):
        super(MountainCar, self).__init__()
        self.episodes = 10000
        self.env = gym.make('MountainCar-v0')
        self.agent = Agent(self.env.action_space.n, self.env.observation_space.shape[0])
        self.is_fit = not os.path.isfile(weight_backup)


    def run(self):
        try:
            lasted_scores = deque(maxlen=20)
            # self.is_fit = True
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1,2])
                done = False
                memorys = []
                index = 0
                finished_p = 0 
                while not done:
                    self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    if done:
                        finished_v = next_state[1]
                        finished_p = next_state[0]
                    next_state = np.reshape(next_state, [1, 2])
                    assert reward==-1, 'reward is not -1'
                    memorys.append((state, action, reward, next_state, done))
                    index += 1
                    state = next_state
                score = 200-index
                [self.agent.remember(*i, score/200+np.abs(finished_p+0.6)) for i in memorys]
                lasted_scores.append(score)
                avg = sum(lasted_scores)/len(lasted_scores)
                print('episode {}# score: {}'.format(index_episode, score))
                self.agent.rethink(self.is_fit)
        finally:
            self.agent.save_model()


if __name__ == '__main__':
    mountainCar = MountainCar()
    mountainCar.run()