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
        self.actor=Model(2,2)
        self.critic=Model(2,2).load("PG.pt")
        self.score=0
        self.tick=0
        self.probs=[]
        self.state=[]
        self.entropy=[]
        self.gaps=[]
        self.done=False

    def select_action(self):
        self.tick+=1
        mu,sigma=self.actor(tensor(self.state).float())
        action, prob, self.entropy=normalsample(mu,sigma)
        c_mu,c_sigma=self.critic(tensor(self.state).float())
        c_action, _, _ =normalsample(c_mu, c_sigma)
        self.gaps.append(c_action.item()-action.item())
        self.probs.append(prob)
        loss=abs(action-c_action)*prob
        self.actor.optimize(loss)
        return array([action.item()])
        

def run():
    env=gym.make("MountainCarContinuous-v0")
    print("observation space:",env.observation_space.low,env.observation_space.high)
    print("action_space:",env.action_space.low,env.action_space.high)
    agent=Agent()
    scope=Scope(Scope.line)
    scores=Recorder(size=100)
    for j in range(1,500):
        agent.state=env.reset()
        agent.score=agent.state[0]
        agent.hit_count=0
        agent.tick=0
        agent.done=False
        while not agent.done:
            action=agent.select_action()
            ob, _, _, _ = env.step(action)
            agent.done=ob[0]>0.45 or agent.tick>998
            agent.score=max(agent.score,ob[0])
            agent.state=ob
        scores.append(agent.score)
        scope.feed(scores.mean(),j)
        print("["+str(j)+"] "+str(agent.score)+" in "+str(agent.tick)+" s "+(str(agent.score) if agent.score>0.45 else ""))
        if j%100==0:
            scores.hist()
    agent.actor.save("IL.pt")
        
time=timeit('run()','from __main__ import run',number=1)
print("Finished after "+str(time)+" s")

