import gym
from rlab import Model,normalsample,factors
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
import visdom
env=gym.make("MountainCarContinuous-v0")
viz=visdom.Visdom()
class Agent():
    def __init__(self):
        self.m=Model(2,2)
        self.probs=[]
        self.score=0
        self.peak=0

    def select_action(self,state):
        mu,sigma=self.m(tensor(state).float())
        a, prob=normalsample(mu,sigma)
        self.probs.append(prob)
        return array([a])

    def optimize(self):
        steps=len(self.probs)
        loss=self.score/steps*-sum(self.probs)
        # print(loss)
        self.m.optimize(loss)
        self.probs=[]
        return loss.item()

agent=Agent()
scores=[]
viz.close(win=None)
win=viz.line([0])

for j in count(1):
    state=env.reset()
    agent.score=state[0]
    agent.peak=0
    for t in range(1000):
        action=agent.select_action(state)
        state, reward, done, _ = env.step(action)
        if agent.score<state[0]:
            agent.score=state[0]
            agent.peak=t
        # env.render()
        if done:
            break
    agent.optimize()
    scores.append(agent.score)
    if j%10==0:   
        s=tensor(scores).mean().item()
        scores.clear()
        viz.line(X=[j],Y=[s],update="append",win=win)
    print("episode "+str(j)+" best "+str(agent.score)+" in "+str(t)+" steps")


