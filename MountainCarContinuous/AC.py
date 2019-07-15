import gym
from rlab import Model,normalsample,Scope,Recorder
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from random import sample,randint
from subprocess import Popen
from time import sleep


class Agent():

    def __init__(self):
        self.actor=Model(2,2)
        self.critic=Model(2,1)
        self.score=0
        self.tick=0
        self.logprob=[]
        self.entropy=[]
        self.state=[]
        self.scores=Recorder(stack_length=100)
        self.done=False

    def select_action(self):
        self.tick+=1
        mu,sigma=self.actor(tensor(self.state).float())
        action, self.prob, self.entropy=normalsample(mu,sigma)
        return array([action])

    def optimize(self,state_):
        self.done=state_[0]>0.45
        r=1 if self.done else 0
        v=self.critic(tensor(self.state).float())
        v_=self.critic(tensor(state_).float()) if not self.done else tensor(0)
        value_loss=abs(r+0.99*v_-v)
        self.critic.optimize(value_loss)
        # -self.prob*(r+0.99*v_.item())
        policy_loss=-self.entropy
        self.actor.optimize(policy_loss)
        

# Popen("visdom")
# sleep(1)
env=gym.make("MountainCarContinuous-v0")
print("observation space:",env.observation_space.low,env.observation_space.high)
print("action_space:",env.action_space.low,env.action_space.high)
agent=Agent()
scope=Scope(Scope.line,10)

for j in count(1):
    agent.state=env.reset()
    agent.score=agent.state[0]
    agent.tick=0
    for t in range(1,1000):
        action=agent.select_action()
        ob, _, _, _ = env.step(action)
        if agent.score<ob[0]:
            agent.score=ob[0]  
        agent.optimize(ob)
        agent.state=ob
        if agent.done:
            break
    agent.scores.record(agent.score)
    scope.feed(agent.scores.mean(),j)
    new_score=" "+str(agent.score) if agent.tick<999 else ""
    print("episode "+str(j)+" best "+str(agent.score)+" in "+str(t)+" steps"+new_score)