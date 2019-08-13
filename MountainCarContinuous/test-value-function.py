import gym
from rlab import *
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from random import sample,randint
from timeit import timeit
from math import e,log

class Agent():

    def __init__(self):
        self.actor=Model(2,2)
        self.score=0
        self.tick=0
        self.prob=[]
        self.state=[]
        self.entropy=[]
        self.done=False
        self.heights=[]

    def select_action(self):
        self.tick+=1
        self.heights.append(self.state[0])
        return array([0.])
        # mu,sigma=self.actor(tensor(self.state).float())
        # action, self.prob, self.entropy=normalsample(mu,sigma)
        # return array([action])

    def reset(self,state):
        self.state=state
        self.score=state[0]
        self.tick=0
        self.done=False
        self.heights=[]

def run():
    env=gym.make("MountainCarContinuous-v0")
    print("observation space:",env.observation_space.low,env.observation_space.high)
    print("action_space:",env.action_space.low,env.action_space.high)
    agent=Agent()
    scores=Recorder(size=10000)
    # heights=Recorder(size=100000)
    for j in range(1,3):
        env._elapsed_steps=0
        env.state=array([-1.,0.])
        agent.reset(env.state)
        while not agent.done:
            action=agent.select_action()
            print(agent.state,action)
            ob, _, _, _ = env.step(action)
            agent.done=ob[0]>0.45 or agent.tick>10
            agent.score=max(agent.score,ob[0])
            agent.state=ob
        scores.append(agent.score)
        plot(agent.heights)
        # plot(negative_max(agent.heights))
        print("["+str(j)+"] "+str(agent.score)+" in "+str(agent.tick)+" steps "+(str(agent.score) if agent.score>0.45 else ""))
        if j%100==0:
            scores.hist()
        
time=timeit('run()','from __main__ import run',number=1)
print("Finished after "+str(time)+" s")