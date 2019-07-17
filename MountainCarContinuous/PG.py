import gym
from rlab import Model,normalsample,Scope,Recorder
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from timeit import timeit

class Agent():
    def __init__(self):
        self.m=Model(2,2)
        self.probs=[]
        self.score=0

    def select_action(self,state):
        mu,sigma=self.m(tensor(state).float())
        a, prob, _=normalsample(mu,sigma)
        self.probs.append(prob)
        return array([a])

    def optimize(self):
        steps=len(self.probs)
        loss=100*self.score/steps*-sum(self.probs)
        self.m.optimize(loss)
        self.probs=[]
        return loss.item()

env=gym.make("MountainCarContinuous-v0")
agent=Agent()
scope=Scope(Scope.line,10)
scores=Recorder(100)

def run():
    average_score=0
    j=0
    while average_score<0.45:
        state=env.reset()
        agent.score=state[0]
        for t in range(1,1000):
            action=agent.select_action(state)
            state, _, done, _ = env.step(action)
            if agent.score<state[0]:
                agent.score=state[0]
            if done:
                break
        agent.optimize()

        scores.append(agent.score)
        average_score=scores.mean()
        scope.feed(average_score,j)
        j+=1
        print("["+str(j)+"] score "+str(agent.score)+" in "+str(t)+" steps "+ (str(agent.score) if agent.score>0.45 else ""))

time=timeit('run()','from __main__ import run',number=1)
print("Finished after "+str(time)+" s")