import gym
from rlab import Model,normalsample,Scope
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from random import sample,randint

class Agent():

    def __init__(self):
        self.m=Model(2,3)
        self.score=0 
        self.prob=[]
        self.state=[]
        self.value=[]

    def select_action(self):
        mu,sigma,self.value=self.m(tensor(self.state).float())
        action, self.prob=normalsample(mu,sigma)
        return array([action])

    def optimize(self,state_):
        _,_,v_=self.m(tensor(state_).float())
        if state_[0]>0.45:
            T=state_[0]*10
        else:
            T=state_[0]+0.5*v_
        loss=-self.prob*(v_-self.value)+abs(T-self.value)
        self.m.optimize(loss)

env=gym.make("MountainCarContinuous-v0")
agent=Agent()
scope=Scope(Scope.line)

for j in count(1):
    agent.state=env.reset()
    agent.score=agent.state[0]
    if j%100==0:
        scope1=Scope(Scope.scatter)
    for t in range(1,1000):
        action=agent.select_action()
        state_, reward, done, _ = env.step(action)
        if agent.score<state_[0]:
            agent.score=state_[0]     
        if j%100==0:
            scope1.feed([agent.state[0],agent.state[1],agent.value.item()])
        agent.optimize(state_)
        agent.state=state_
        if done:
            break
    scope.feed(agent.score,j,10)
    print("episode "+str(j)+" best "+str(agent.score)+" in "+str(t)+" steps")
    # print(agent.m(tensor([0.45,0]))[2].item())