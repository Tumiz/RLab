from rlab import negtive_max, plot
from torch import tensor, rand

a=rand(10)
print(a)
a_nm=negtive_max(a)
print(a_nm)
plot(a_nm)

b=a.tolist()
print(b)
b_nm=negtive_max(b)
print(b_nm)
plot(b_nm)

