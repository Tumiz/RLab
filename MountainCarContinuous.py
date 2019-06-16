import gym
from base import Model,normalsample,totalvalue,calvalues
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
        self.rewards=[]
        self.probs=[]
        self.optimizer=Adam(self.m.parameters(), lr=0.01)

    def select_action(self,state):
        mu,sigma=self.m(tensor(state).float())
        a, prob=normalsample(mu,sigma)
        self.probs.append(prob)
        # print(prob)
        return array([a])

    def optimize(self):
        loss=calvalues(self.rewards,gamma=0.99,normalized=True).sum()*sum(self.probs)
        # print(loss)
        self.m.optimize(loss)
        self.rewards=[]
        self.probs=[]
        return loss.item()

agent=Agent()
scores=[]
losses=[]
viz.close(win=None)
win=viz.line([0])

for j in count(1):
    state=env.reset()
    bestscore=state[0]
    besttime=0
    for i in range(1000):
        action=agent.select_action(state)
        state, reward, done, _ = env.step(action)
        if bestscore<state[0]:
            bestscore=state[0]
            besttime=i
        reward=linalg.norm(state)
        agent.rewards.append(reward)
        # env.render()
        if done:
            break
    # agent.rewards=zeros(besttime+1)
    # agent.rewards[besttime]=bestscore/besttime
    # agent.probs=agent.probs[0:besttime+1]
    loss=agent.optimize() 
    scores.append(bestscore)
    losses.append(loss)
    if j%10==0:   
        s=tensor(scores).mean().item()
        l=tensor(losses).mean().item()
        scores.clear()
        losses.clear()
        viz.line(X=[j],Y=[s],update="append",win=win)
    print("episode "+str(j)+" best "+str(bestscore)+" in "+str(i)+" steps")


