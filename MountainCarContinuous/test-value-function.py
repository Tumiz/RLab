from itertools import count
from math import e, log
from random import randint, sample
from timeit import timeit

from numpy import array, linalg, zeros
from torch import tensor,empty
from torch.optim import Adam
from torch.nn import L1Loss

import gym
from rlab import *


class Agent():

    def __init__(self):
        self.actor = Model(2, 2).init_to(0.8)
        self.critic=Model(3,1,10,6)
        self.score = 0
        self.tick = 0
        self.prob = []
        self.state = []
        self.entropy = []
        self.done = False
        self.obs=[]
        self.values=[]
        self.test_chart=Chart()

    def select_action(self):
        self.tick += 1
        mu, sigma = self.actor(tensor(self.state).float())
        action, self.prob, self.entropy = normalsample(mu, sigma)
        return array([action])

    def reset(self, ob):
        self.state = ob
        self.score = ob[0]
        self.tick = 0
        self.done = False
        self.obs=[ob]

    def collect_data(self,ob):
        self.obs.append(ob)
        r=self.obs[::-1]
        m=r[0][0]
        n=0
        for i in r:
            m=max(m,i[0])
            self.values.append([i[0],i[1],n,m])
            n+=1

    def on_finished(self):
        if len(self.values)>100000:
            self.values=sample(self.values,100000)
        data=tensor(self.values)
        inputs=data[:,0:3]
        outputs=data[:,3]
        predictions=self.critic(inputs)
        loss_func=L1Loss()
        loss=loss_func(outputs,predictions.reshape(len(predictions)))
        self.critic.optimize(loss)

    def value_test(self):
        inputs=empty(999,3)
        inputs[:,0]=-1
        inputs[:,1]=0
        inputs[:,2]=tensor(range(999))
        self.test_chart.plot(self.critic(inputs))

def env_reset_to(env,state):
    env._elapsed_steps = 0
    env.env.state = state

def run():
    env = gym.make("MountainCarContinuous-v0")
    print("observation space:", env.observation_space.low,
          env.observation_space.high)
    print("action_space:", env.action_space.low, env.action_space.high)
    agent = Agent()
    scores = Recorder(size=10000)
    chart_heights= Chart()
    # heights=Recorder(size=100000)
    for j in range(1, 100):
        agent.reset(env.reset())
        while not agent.done:
            action = agent.select_action()
            ob, _, _, _ = env.step(action)
            agent.collect_data(ob)
            agent.done = ob[0] > 0.45 or agent.tick > 998
            agent.score = max(agent.score, ob[0])
            agent.state = ob
        agent.on_finished()
        if j>50:
            agent.value_test()
        scores.append(agent.score)

        print("["+str(j)+"] "+str(agent.score)+" in "+str(agent.tick) +
              " steps "+(str(agent.score) if agent.score > 0.45 else ""))
        if j % 100 == 0:
            scores.hist()

    


time = timeit('run()', 'from __main__ import run', number=1)
print("Finished after "+str(time)+" s")
