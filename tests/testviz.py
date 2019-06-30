from rlab import Scope
from torch import tensor
scope=Scope(Scope.scatter)
scope.feed([0,1,2],samplerate=2)
scope.feed([4,5,6],samplerate=2)
scope.feed([7,8,6],samplerate=2)