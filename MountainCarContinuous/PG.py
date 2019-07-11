import gym
from rlab import Model,normalsample,Scope
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count

class Agent():
    def __init__(self):
        self.m=Model(2,2)
        self.probs=[]
        self.score=0

    def select_action(self,state):
        mu,sigma=self.m(tensor(state).float())
        a, prob=normalsample(mu,sigma)
        self.probs.append(prob)
        return array([a])

    def optimize(self):
        steps=len(self.probs)
        loss=100*self.score*-sum(self.probs)
        self.m.optimize(loss)
        self.probs=[]
        return loss.item()

env=gym.make("MountainCarContinuous-v0")
agent=Agent()
scope=Scope(Scope.line)

for j in count(1):
    state=env.reset()
    agent.score=state[0]
    for t in range(1,1000):
        action=agent.select_action(state)
        state, reward, done, _ = env.step(action)
        if agent.score<state[0]:
            agent.score=state[0]
        if done:
            break
        agent.optimize()
    scope.feed(agent.score,j,10)
    print("episode "+str(j)+" best "+str(agent.score)+" in "+str(t)+" steps")