import gym
from rlab import Model,normalsample,Scope,Recorder
from torch import tensor
from torch.optim import Adam
from numpy import array,zeros,linalg
from itertools import count
from random import sample,randint
from timeit import timeit

class Agent():

    def __init__(self):
        self.m=Model(2,3)
        self.score=0
        self.tick=0
        self.prob=[]
        self.state=[]
        self.done=False

    def select_action(self):
        self.tick+=1
        mu,sigma,_=self.m(tensor(self.state).float())
        action, self.prob, _=normalsample(mu,sigma)
        return array([action])

    def optimize(self,state_):
        _,_,v=self.m(tensor(self.state).float())
        if not self.done:
            _,_,v_=self.m(tensor(state_).float()) 
        else:
            v_=tensor(0.)
        td=max(state_[0],v_)-v
        value_loss=-max(tensor(state_[0]),v)
        self.m.optimize(value_loss)
        

def run():
    env=gym.make("MountainCarContinuous-v0")
    print("observation space:",env.observation_space.low,env.observation_space.high)
    print("action_space:",env.action_space.low,env.action_space.high)
    agent=Agent()
    scope=Scope(Scope.line)
    scores=Recorder(stack_length=5)
    average_score=0
    j=0
    while average_score<0.45:
        j+=1
        agent.state=env.reset()
        agent.score=agent.state[0]
        agent.tick=0
        while True:
            action=agent.select_action()
            ob, _, _, _ = env.step(action)
            agent.score=max(agent.score,ob[0])
            agent.done=agent.score>0.45 or agent.tick>998
            agent.optimize(ob)
            agent.state=ob
            if agent.done:
                break
        scores.append(agent.score)
        average_score=scores.mean()
        print("["+str(j)+"] "+str(agent.score)+" in "+str(agent.tick)+" steps "+(">0.45 " if agent.score>0.45 else "")+str(round(average_score,2)))
        scope.feed(average_score,j)
        
time=timeit('run()','from __main__ import run',number=1)
print("Finished after "+str(time)+" s")