import sys; sys.path.append('../')

import numpy as np
import numba as nb
import nbstl, random
from time import time

# custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# TypedMemory cast Memory as dtype
FloatAVL = nbstl.TypedAVLTree(np.float32)
lst = FloatAVL()

x = np.random.rand(10000000).astype(np.float32)

@nb.njit
def insert(lst, x):
    for i in x: lst.push(i)

@nb.njit
def pop(lst, x):
    for i in x: lst.pop(i)

insert(lst, x[:1])
pop(lst, x[:1])

lst = FloatAVL()
a = time()
insert(lst, x)
b = time()
pop(lst, x)
print('1000w number avl insert %.3fs del %.3f s:'%(b-a, time()-b))

# TypedMemory cast Memory as dtype
FloatRB = nbstl.TypedRBTree(np.float32)
lst = FloatRB()

insert(lst, x[:1])
pop(lst, x[:1])

lst = FloatRB()
a = time()
insert(lst, x)
b = time()
pop(lst, x)
print('1000w number redblack insert %.3fs del %.3f s:'%(b-a, time()-b))
