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

x = np.random.rand(10000000)

@nb.njit
def insert(lst, x):
    for i in x:
        lst.push(i)

insert(lst, x[:1])

start = time()
lst = FloatAVL()
insert(lst, x)
print('avl cost', time()-start)

# TypedMemory cast Memory as dtype
FloatRB = nbstl.TypedRBTree(np.float32)
lst = FloatRB()

insert(lst, x[:1])

start = time()
lst = FloatRB()
insert(lst, x)
print('redblack cost', time()-start)
