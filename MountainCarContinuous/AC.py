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
        self.actor=Model(2,2)
        self.critic=Model(2,1)
        self.score=0
        self.tick=0
        self.prob=[]
        self.state=[]
        self.done=False

    def select_action(self):
        self.tick+=1
        mu,sigma=self.actor(tensor(self.state).float())
        action, self.prob, _=normalsample(mu,sigma)
        return array([action])

    def optimize(self,state_):
        self.done=state_[0]>0.45 or self.tick>998
        v=self.critic(tensor(self.state).float())
        v_=self.critic(tensor(state_).float()) if not self.done else tensor(0)
        td=max(state_[0],0.99*v_)-v
        value_loss=abs(td)
        self.critic.optimize(value_loss)
        policy_loss=-self.prob*td.item()
        self.actor.optimize(policy_loss)
        

def run():
    env=gym.make("MountainCarContinuous-v0")
    print("observation space:",env.observation_space.low,env.observation_space.high)
    print("action_space:",env.action_space.low,env.action_space.high)
    agent=Agent()
    scope=Scope(Scope.line,10)
    scores=Recorder(stack_length=100)
    average_score=0
    for j in count(1):
        agent.state=env.reset()
        agent.score=agent.state[0]
        agent.tick=0
        while True:
            action=agent.select_action()
            ob, _, _, _ = env.step(action)
            agent.score=max(agent.score,ob[0])
            agent.optimize(ob)
            agent.state=ob
            if agent.done:
                break
        scores.record(agent.score)
        average_score=scores.mean()
        scope.feed(average_score,j)
        print("["+str(j)+"] "+str(agent.score)+" in "+str(agent.tick)+" steps "+(str(agent.score) if agent.score>0.45 else ""))

time=timeit('run()','from __main__ import run',number=1)
print("Finished after "+str(time)+" s")