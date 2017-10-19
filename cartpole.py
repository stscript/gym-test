import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

weight_backup      = "cartpole_weight.h5"

class Agent():
    def __init__(self, state_size, action_size):
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.96
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        # model.add(Dense(32, activation='relu'))
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
            # print('random')
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done, score=1):
        self.memory.append((state, action, reward, next_state, done, score/500))

    def replay(self, sample_batch_size, is_fit=True):
        # print(is_fit)
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done, score in sample_batch:
            target = reward + score
            if not done:
              p = self.brain.predict(next_state)
              target = reward + score + self.gamma * np.amax(p[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            if is_fit:
                self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000
        self.env               = gym.make('CartPole-v0')
        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)
        self.is_fit            = not os.path.isfile(weight_backup)
        self.monitor_location  = '/tmp/cartpole-v0' 

    
    def test(self): 
        self.env = gym.wrappers.Monitor(self.env, self.monitor_location, force=True)
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])

        for t in range(200):
            self.env.render()
            action = self.agent.act(state)
            state, reward, done, _ = self.env.step(action)
            state = np.reshape(state, [1, self.state_size])
            if done:
                print('Episode finished after {} times'.format(t+1))
                self.env.reset()
                break
        self.env.close()


    def upload(self):
        gym.upload(self.monitor_location, api_key='c84cac95d89232cefde70a676ab183a2c00843c8')


    def run(self):
        try:
            lasted_scores = deque(maxlen=20)

            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                memorys = []
                while not done:
                    self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    assert reward == 1, 'reward is %d'%reward
                    memorys.append((state, action, reward, next_state, done))
                    state = next_state
                    index += 1
                [self.agent.remember(*i, index+1) for i in memorys]
                lasted_scores.append(index)
                avg = sum(lasted_scores)/len(lasted_scores)
                # print(avg)
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                if avg>480:
                    self.is_fit = False
                self.agent.replay(self.sample_batch_size, self.is_fit)
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    cartpole = CartPole()
    # cartpole.test()
    cartpole.upload()