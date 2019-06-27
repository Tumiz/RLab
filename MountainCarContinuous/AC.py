import gym
from rlab import Model,normalsample
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from random import sample,randint
import visdom

env=gym.make("MountainCarContinuous-v0")
viz=visdom.Visdom()

class Agent():

    def __init__(self):
        self.actor=Model(2,2)
        self.critic=Model(3,1)
        self.score=0 
        self.prob=[]
        self.state=[]
        self.action=[]
        self.reward=[]
        self._state=[]
        self._action=[]
        self._reward=[]
        self.memory=[]

    def select_action(self):
        mu,sigma=self.actor(tensor(self.state).float())
        self.action, self.prob=normalsample(mu,sigma)
        return array([self.action])

    def optimize(self):
        if len(self._state):
            self.memory.append([self._state,self._action,self._reward,self.state,self.action])
        q=self.critic(tensor([self.state[0],self.state[1],self.action]))
        loss=self.prob*q
        self.actor.optimize(loss)
        l=len(self.memory)
        if l:
            i=randint(0,l-1)
            sa=self.memory[i]
            s,a,r,s_,a_=sa
            T=r+0.99*self.critic(tensor([s_[0],s_[1],a_]))
            R=self.critic(tensor([s[0],s[0],a]))
            loss=abs(T-R)
            self.critic.optimize(loss)
            if l>1000:
                self.memory.pop(i)

agent=Agent()
scores=[]
viz.close(win=None)
win=viz.line([0])

for j in count(1):
    agent.state=env.reset()
    agent.score=agent.state[0]
    for t in range(1,1000):
        action=agent.select_action()
        state_, reward, done, _ = env.step(action)
        if agent.score<state_[0]:
            agent.score=state_[0]
        agent.reward=state_[0]-agent.state[0]
        agent.optimize()
        agent._state=agent.state
        agent._action=agent.action
        agent._reward=agent.reward
        agent.state=state_
        if done:
            break
    scores.append(agent.score)
    if j%10==0:   
        s=tensor(scores).mean().item()
        scores.clear()
        viz.line(X=[j],Y=[s],update="append",win=win)
    print("episode "+str(j)+" best "+str(agent.score)+" in "+str(t)+" steps")


