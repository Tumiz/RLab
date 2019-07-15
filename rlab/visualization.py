import visdom
from torch import tensor

class Scope():
    line=1
    scatter=2
    def __init__(self,scope_type=line,sample_rate=1):
        self.type=scope_type
        self.tick=0
        self.sample_rate=sample_rate
        self.data=[]
        self.stamps=[]
        self.viz=visdom.Visdom()
        self.win=self.viz.line(Y=[0],X=[0])
    
    def feed(self,d,t=-1):
        self.tick+=1
        if self.tick%self.sample_rate==0:
            self.data.append(d)
            if t>=0:
                self.stamps.append(t)
            else:
                self.stamps.append(len(self.stamps))
            if self.type==Scope.line:
                self.viz.line(Y=self.data,X=self.stamps,win=self.win)
            elif self.type==Scope.scatter:
                self.viz.scatter(self.data,win=self.win,opts=dict(markersize=3))

    def reset(self):
        self.data.clear()
        self.stamps.clear()
        self.tick=0



