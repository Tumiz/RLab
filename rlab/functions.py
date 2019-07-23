from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh, rand
from torch.nn import Module, Linear, MSELoss
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from random import random
from math import cos, sin, tan, pi, floor
from visdom import Visdom

# viz = Visdom()


def normalize(data):  # input tensor
    ret = (data-data.mean())/(data.std()+1e-10)
    return ret


def gather(feeds, gamma=0.99):  # input list
    values = []
    V = 0
    for feed in feeds[::-1]:
        V = feed+gamma*V
        values.insert(0, V)
    return values

def calvalues(rewards, gamma=0.99, normalized=False):
    values=gather(rewards,gamma)
    if normalized:
        values=normalize(tensor(values))
    return values

def totalvalue(rewards, gamma=0.99):
    return calvalues(rewards, gamma).sum()

def processlogprob(logprobs):
    return sum(logprobs)

def truncatedsample(samplefunc, low, high):
    sample = samplefunc()
    while sample <= low or sample >= high:
        sample = samplefunc()
    return sample


def normalsample(mu, sigma):
    dist = Normal(mu, softplus(sigma))
    sample = dist.sample().item()
    logprob = dist.log_prob(sample)
    return sample, logprob, dist.entropy()

def wraptopi(x):
    x = x - floor(x/(2*pi)) *2 *pi
    if(x>pi and x<2*pi):
        x=x-2*pi
    if x<-pi and x>-2*pi:
        x=2*pi+x
    return x

def factors(length,peak):
    ret=[]
    for i in range(length):
        if i<peak:
            ret.append(pow(0.99,peak-i))
        elif i==peak:
            ret.append(1)
        else:
            ret.append(pow(0.99,i-peak))
    return ret

class Model(Module):
    def __init__(self, nin, nout):
        super(Model, self).__init__()
        self.layer1 = Linear(nin, nin*nout*4)
        self.layer2 = Linear(nin*nout*4, nout)
        self.optimizer = Adam(self.parameters())

    def forward(self, x):
        x = elu(self.layer1(x))
        return self.layer2(x)

    def optimize(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def optimize1(self,a,b):
        loss_func = MSELoss(reduce = True,size_average = True)
        loss=loss_func(self(a),b)
        self.optimize(loss)


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

    def step(self, t):
        dt=t-self.t
        vx = self.v*cos(self.h)
        vy = self.v*sin(self.h)
        self.x += vx*dt
        self.y += vy*dt
        self.v += self.a*dt
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
        self.car.x=-5
        self.car.y=(random()-1)*5
        self.car.h=random()*pi*0.5
        self.state=self.car.state()
        self.state_distance=self.state.norm()
        self.done = False

    def step(self, a, w):
        self.car.a = a
        self.car.w = w
        self.car.step(self.t)
        self.t += 0.01
        self.state = self.car.state()
        self.state_distance=self.state.norm()
        if(self.t>0.5 or self.state_distance>30):
            self.done=True
            return self.state_distance
        else:
            return 0

class Recorder:
    def __init__(self,stack_length=100,sample_rate=1):
        self.stack=[]
        self.stack_length=stack_length
        self.sample_rate=sample_rate
        self.tick=0

    def append(self,data):
        self.tick+=1
        if self.tick%self.sample_rate==0:
            self.stack.append(data)
            if len(self.stack)>self.stack_length:
                self.stack.pop(0)
                self.tick=0

    def mean(self,n=100):# n should not be larger than stack_length
        n=min(n,len(self.stack))
        return tensor(self.stack[-n:]).mean().item() if n>0 else tensor(0.)

    def max(self):
        return tensor(self.stack).max().item()

    def size(self):
        return len(self.stack)

    def clear(self):
        self.stack=[]
        self.tick=0