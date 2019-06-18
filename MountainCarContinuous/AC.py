import gym
from base import Model,normalsample,totalvalue,gather
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from random import randint
import visdom

env=gym.make("MountainCarContinuous-v0")
viz=visdom.Visdom()

class Agent():

    def __init__(self):
        self.actor=Model(2,2)
        self.critic=Model(2,1)
        self.states=[]
        self.rewards=[]
        self.probs=[]
        self.memory_state=[]
        self.memory_value=[]

    def select_action(self,state):
        mu,sigma=self.actor(tensor(state).float())
        a, prob=normalsample(mu,sigma)
        self.probs.append(prob)
        self.states.append(state)
        # print(prob)
        return array([a])

    def update_by_episode(self):
        values=gather(self.rewards,gamma=0.99)
        # loss=sum(values)*sum(self.probs) 
        # self.actor.optimize(loss)
        if len(self.memory_state)>1000:
            self.memory_state.pop(randint(0,1000))
            self.memory_value.pop(randint(0,1000))
        self.memory_state+=self.states
        self.memory_value+=values
        self.states=[]
        self.rewards=[]
        self.probs=[]

    def update_by_step(self,state_):
        state=self.states[-1]
        prob=self.probs[-1]
        value=self.critic(tensor(state).float())
        value_=self.critic(tensor(state_).float())
        loss=prob*(value-value_)
        self.actor.optimize(loss)
        if len(self.memory_state)>0:
            self.critic.optimize1(tensor(self.memory_state).float(),tensor(self.memory_value).float().reshape(len(self.memory_state),1))

agent=Agent()
scores=[]
viz.close(win=None)
win=viz.line([0])

for j in count(1):
    state=env.reset()
    bestscore=state[0]
    besttime=0
    for i in range(1000):
        action=agent.select_action(state)
        state, reward, done, _ = env.step(action)
        agent.update_by_step(state)
        if bestscore<state[0]:
            bestscore=state[0]
            besttime=i
        reward=linalg.norm(state)
        agent.rewards.append(reward)
        # env.render()
        if done:
            break
    agent.update_by_episode()
    scores.append(bestscore)
    if j%10==0:   
        s=tensor(scores).mean().item()
        scores.clear()
        viz.line(X=[j],Y=[s],update="append",win=win)
    print("episode "+str(j)+" best "+str(bestscore)+" in "+str(i)+" steps")


