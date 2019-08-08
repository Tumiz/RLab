import gym
from rlab import Model,normalsample,Scope,Recorder
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from random import sample,randint
from timeit import timeit
from math import e,log

class Agent():

    def __init__(self):
        self.actor=Model(2,2).load("IL.pt")
        self.score=0
        self.tick=0
        self.prob=[]
        self.state=[]
        self.entropy=[]
        self.done=False

    def select_action(self):
        self.tick+=1
        if self.tick<100:
            return array([0.03])
        mu,sigma=self.actor(tensor(self.state).float())
        action, self.prob, self.entropy=normalsample(mu,sigma)
        return array([action])
        

def run():
    env=gym.make("MountainCarContinuous-v0")
    print("observation space:",env.observation_space.low,env.observation_space.high)
    print("action_space:",env.action_space.low,env.action_space.high)
    agent=Agent()
    scope=Scope(Scope.line)
    scores=Recorder(size=10000)
    # heights=Recorder(size=100000)
    j=0
    while True:
        j+=1
        agent.state=env.reset()
        agent.score=agent.state[0]
        agent.tick=0
        agent.done=False
        while not agent.done:
            action=agent.select_action()
            ob, _, _, _ = env.step(action)
            agent.done=ob[0]>0.45 or agent.tick>998
            agent.score=max(agent.score,ob[0])
            # agent.optimize(ob)
            agent.state=ob
            heights.append(ob[0])
        scores.append(agent.score)
        scope.feed(scores.mean(),j)
        print("["+str(j)+"] "+str(agent.score)+" in "+str(agent.tick)+" steps "+(str(agent.score) if agent.score>0.45 else ""))
        # if j%100==0:
        #     heights.hist()
        if j%100==0:
            scores.hist()
time=timeit('run()','from __main__ import run',number=1)
print("Finished after "+str(time)+" s")