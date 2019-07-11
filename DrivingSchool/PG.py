from rlab import Model,normalsample,Scope,wraptopi
from torch import tensor
from torch.optim import Adam
from torch.nn.functional import softplus
from math import cos,tan,sin,pi,tanh
from numpy import array,zeros,linalg
from itertools import count
from random import sample,randint,random


class Car:
    def __init__(self):
        self.a = 0.
        self.w = 0.
        self.t = 0.
        self.v = 0.
        self.x = 0.
        self.y = 0.
        self.h = 0.
        # self.plot=viz.line(X=[0],Y=[[self.a,self.v,self.x]])

    def step(self, dt):
        vx = self.v*cos(self.h)
        vy = self.v*sin(self.h)
        self.x += vx*dt
        self.y += vy*dt
        self.v += self.a*dt
        self.v = max(-10,self.v)
        self.v = min(50,self.v)
        self.h += self.v/2.7*tan(self.w)*dt
        self.h = wraptopi(self.h)
        # viz.line(X=[t],Y=[[self.a,self.v,self.x]],win=self.plot,update='append')

    def reset(self):
        self.a=0.
        self.w=0.
        self.v=0.
        self.x=0.
        self.y=0.
        self.h=0.
        self.t=0.

    def state(self):
        return tensor([self.v,self.x,self.y,self.h])


class Environment:
    def __init__(self):
        self.t = 0.
        self.car = Car()
        self.car.x=-50
        self.car.y=-50
        self.car.h=0
        self.state=self.car.state()
        self.state_distance=self.state.norm().item()
        self.done = False

    def step(self, a, w):
        self.car.a = a
        self.car.w = w
        self.car.step(0.01)
        self.t += 0.01
        self.state = self.car.state()
        self.state_distance=self.state.norm().item()

class Agent():
    def __init__(self):
        self.m=Model(4,4)
        self.probs=[]
        self.score=0

    def select_action(self,state):
        a_mu,a_sigma,w_mu,w_sigma=self.m(state)
        a_mu,a_sigma,w_mu,w_sigma=a_mu,softplus(a_sigma),w_mu*0.6,softplus(w_sigma)
        a, a_prob=normalsample(a_mu,a_sigma)
        w, w_prob=normalsample(w_mu,w_sigma)
        self.probs.append(a_prob+w_prob)
        return tanh(a),tanh(w)*0.6

    def optimize(self):
        steps=len(self.probs)
        loss=self.score*steps*sum(self.probs)/10000000
        self.m.optimize(loss)
        self.probs=[]
        return loss.item()


class Simulation:

    def __init__(self):
        self.env = Environment()
        self.agent = Agent()

    def run(self,times,plotinterval=1000):
        for episode in range(0,times):
            rewards=[]
            probs=[]
            actions = []
            states=[]
            self.env.__init__()
            while self.env.state_distance>=1 and self.env.t<20:
                a, w = self.agent.select_action(self.env.state)
                self.env.step(a,w)
            self.agent.score=self.env.state_distance
            self.agent.optimize()
            print(str(episode)+" "+str(self.agent.score)+" in "+str(self.env.t)+" s")

sim=Simulation()
sim.run(1000)
