import sys; sys.path.append('../')

import numpy as np
import numba as nb
import structronref, random
from numba.experimental import structref
from time import time

__name__ = 'avlrbtreereftest'
sys.modules['avlrbtreereftest'] = sys.modules.pop('__main__')

# custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# TypedMemory cast Memory as dtype
class FloatAVLStruct(nb.types.StructRef): pass
class FloatAVLProxy(structref.StructRefProxy): pass
FloatAVL = structronref.TypedAVLTree(FloatAVLStruct, FloatAVLProxy, None, np.float32)



# TypedMemory cast Memory as dtype
class FloatRBStruct(nb.types.StructRef): pass
class FloatRBProxy(structref.StructRefProxy): pass
FloatRB = structronref.TypedRBTree(FloatRBStruct, FloatRBProxy, None, np.float32)



lst = FloatAVL()

x = np.random.rand(10000).astype(np.float32)

@nb.njit(cache=True)
def insert(lst, x):
    for i in x: lst.push(i)

@nb.njit(cache=True)
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


lst = FloatRB()

insert(lst, x[:1])
pop(lst, x[:1])

lst = FloatRB()
a = time()
insert(lst, x)
b = time()
pop(lst, x)
print('1000w number redblack insert %.3fs del %.3f s:'%(b-a, time()-b))

