import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core.extending import overload_method

import sys
__name__ = 'demo'
sys.modules[__name__] = sys.modules.pop('__main__')

@nb.njit(cache=True)
def _add_one(self):
    self.a += 1

class BaseStruct(nb.types.StructRef): pass
class Heap(structref.StructRefProxy): pass

def f():
    struct = structref.register(BaseStruct)
    t_heap = struct([('a', nb.int32)])

    @nb.njit(cache=True)
    def init():
        self = structref.new(t_heap)
        self.a = 0
        return self

    def __new__(cls):
        return init()

    Heap.__new__ = __new__
    Heap.add_one = _add_one
    
    structref.define_boxing(struct, Heap)
    
    return init

from time import time

Heap = f()

@nb.njit
def new():
    return Heap()


aaaa
Heap = f()
start = time()
heap = Heap()
print('new cost', time()-start)



start = time()
heap.add_one()
# _add_one(heap)
print('add cost', time()-start)
