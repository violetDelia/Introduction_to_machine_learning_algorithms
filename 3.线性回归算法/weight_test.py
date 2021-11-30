import numpy as np

narray = np.ones((5,5))
print(narray)
v = [1,2,3,4,5]
v = np.array(v).reshape(1,-1)
print(narray*v)