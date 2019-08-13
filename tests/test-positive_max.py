from rlab import positive_max, plot
from torch import tensor, rand

a=rand(10)
print(a)
a_nm=positive_max(a)
print(a_nm)
plot(a_nm)

b=a.tolist()
print(b)
b_nm=positive_max(b)
print(b_nm)
plot(b_nm)