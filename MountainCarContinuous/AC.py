import gym
from rlab import Model,normalsample,Scope
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from random import sample,randint
from subprocess import Popen
from time import sleep


class Agent():

    def __init__(self):
        self.m=Model(2,3)
        self.score=0
        self.bestscore=-1.2
        self.prob=[]
        self.state=[]
        self.value=[]

    def select_action(self):
        mu,sigma,self.value=self.m(tensor(self.state).float())
        action, self.prob=normalsample(mu,sigma)
        return array([action])

    def step_optimize(self,state_):
        _,_,v_=self.m(tensor(state_).float())
        if state_[0]>0.45:
            r=100
        else:
            r=0
        value_loss=abs(r +v_-self.value)
        loss=-self.prob*(v_-1)+value_loss
        self.m.optimize(loss)

Popen("visdom")
sleep(1)
env=gym.make("MountainCarContinuous-v0")
print("observation space:",env.observation_space.low,env.observation_space.high)
print("action_space:",env.action_space.low,env.action_space.high)
agent=Agent()
scope=Scope(Scope.line)

for j in count(1):
    agent.state=env.reset()
    agent.score=agent.state[0]
    for t in range(1,1000):
        action=agent.select_action()
        ob, reward, done, _ = env.step(action)
        if agent.score<ob[0]:
            agent.score=ob[0]  
        agent.step_optimize(ob)
        agent.state=ob
        if done:
            break
    if agent.bestscore<agent.score:
        agent.bestscore=agent.score
    scope.feed(agent.score,j,10)
    print("episode "+str(j)+" best "+str(agent.score)+"/" + str(agent.bestscore)+" in "+str(t)+" steps")