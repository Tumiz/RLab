import visdom
from torch import tensor
from enum import Enum

class Scope():
    line=1
    scatter=2
    def __init__(self,scope_type):
        self.type=scope_type
        self.temp=[]
        self.data=[]
        self.stamps=[]
        self.viz=visdom.Visdom()
        self.win=self.viz.line(Y=[0],X=[0])
    
    def feed(self,d,t=-1,samplerate=1):
        if len(self.temp)<samplerate-1:
            self.temp.append(d)
        else:  
            if self.type==Scope.line:
                self.data.append(tensor(self.temp).mean().item())
            elif self.type==Scope.scatter:
                self.data.append(d)
            self.temp.clear()
            if t>=0:
                self.stamps.append(t)
            else:
                self.stamps.append(len(self.stamps))
            if self.type==Scope.line:
                self.viz.line(Y=self.data,X=self.stamps,win=self.win)
            elif self.type==Scope.scatter:
                self.viz.scatter(self.data,win=self.win,opts=dict(markersize=3))

    def reset(self):
        self.temp.clear()
        self.data.clear()
        self.stamps.clear()

class Map():
    def __init__(self, *args, **kwargs):
        self.x=[-1,1]
        self.y=[-1,1]


