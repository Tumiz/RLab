from torch.optim import Adam
from torch import tensor, arange, stack, isnan, tanh, rand, save, load
from torch.nn import Module, Linear, MSELoss
from torch.nn.functional import softplus, elu
from torch.distributions.normal import Normal
from random import random
from math import cos, sin, tan, pi, floor
from visdom import Visdom

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

    def init_to(self, value):
        self.layer2.weight.data.fill_(0)
        self.layer2.bias.data.fill_(value)
        return self

    def save(self,path):
        save(self.state_dict(),path)

    def load(self,path):
        self.load_state_dict(load(path))
        self.eval()
        self.optimizer=Adam(self.parameters())
        return self

class Recorder:
    def __init__(self,size=100,sample_rate=1):
        self.stack=[]
        self.size=size
        self.sample_rate=sample_rate
        self.tick=0

    def append(self,data):
        self.tick+=1
        if self.tick%self.sample_rate==0:
            self.stack.append(data)
            if len(self.stack)>self.size:
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

    def hist(self):
        viz=Visdom()
        viz.histogram(self.stack)

    def plot(self):
        viz=Visdom()
        viz.line(self.stack)

class Scope():
    line=1
    scatter=2
    def __init__(self,scope_type=line,sample_rate=1,name=""):
        self.type=scope_type
        self.name=name
        self.tick=0
        self.sample_rate=sample_rate
        self.data=[]
        self.stamps=[]
        self.viz=Visdom()
        self.win=self.viz.line(Y=[0],X=[0],opts=dict(title=name))
    
    def feed(self,d,t=-1):
        self.tick+=1
        if self.tick%self.sample_rate==0:
            self.data.append(d)
            if t>=0:
                self.stamps.append(t)
            else:
                self.stamps.append(len(self.stamps))
            if self.type==Scope.line:
                self.viz.line(Y=self.data,X=self.stamps,win=self.win,opts=dict(title=self.name))
            elif self.type==Scope.scatter:
                self.viz.scatter(self.data,win=self.win,opts=dict(markersize=3,title=self.name))

    def reset(self):
        self.data.clear()
        self.stamps.clear()
        self.tick=0