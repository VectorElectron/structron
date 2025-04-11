import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core.extending import overload_method

@nb.njit(cache=True)
def expand(self):
    self.cont = np.concatenate((self.cont, self.cont))
    self.idx = self.cont['idx']
    self.body = self.cont['body']
    self.idx[self.cap:] = np.arange(self.cap+1, self.cap*2+1, dtype=np.int32)
    self.cur = self.cap
    self.cap *= 2
    # self.tail = self.cap - 1

@nb.njit(cache=True)
def push(self, obj):
    if self.size == self.cap:
        self.expand()
    self.size += 1
    cur = self.cur
    self.cur = self.idx[cur]
    self.body[cur] = obj
    return cur

@nb.njit(cache=True)
def __getitem__(self, idx):
    return self.body[idx]

@nb.njit(cache=True)
def __len__(self):
    return self.size

@nb.njit(cache=True)
def pop(self, idx):
    if self.size==self.cap:
        self.cur = idx
    self.size -= 1
    self.idx[idx] = self.cur
    self.cur = idx
    # self.idx[self.tail] = idx
    # self.tail = idx
    return self.body[idx]

def TypedMemory(BaseStruct, BaseProxy, dtype):
    struct = structref.register(BaseStruct)
    
    idxtype = np.dtype([('idx', np.int32), ('body', dtype)])
    t_memory = struct([('cont', nb.from_dtype(idxtype)[:]),
                ('idx', nb.int32[:]), ('cur', nb.int32),
                ('cap', nb.uint32), ('size', nb.uint32),
                ('body', nb.from_dtype(dtype)[:])])

    temp = ''' # add attrs
        def get_%s(self): return self.%s
        def set_%s(self, v): self.%s = v
        BaseProxy.%s = property(nb.njit(get_%s), nb.njit(set_%s))
        del get_%s, set_%s'''
    temp = '\n'.join([i.strip() for i in temp.split('\n')])

    for i in ('idx', 'body', 'cap', 'cur', 'size'): exec(temp%((i,)*9))

    BaseProxy.push = push
    BaseProxy.pop = pop
    BaseProxy.__getitem__ = __getitem__
    BaseProxy.__len__ = __len__
    
    structref.define_boxing(struct, BaseProxy)

    overload_method(struct, "push")(lambda self, obj: push.py_func)
    overload_method(struct, "expand")(lambda self: expand.py_func)
    overload_method(struct, "pop")(lambda self, idx: pop.py_func)
    
    @nb.njit(cache=True)
    def init(cap=16):
        self = structref.new(t_memory)    
        self.cont = np.empty(cap, dtype=idxtype)
        self.idx = self.cont['idx']
        self.body = self.cont['body']
        self.idx[:] = np.arange(1, cap+1, dtype=np.int32)
        self.cur = 0 # next blank
        self.cap = cap
        self.size = 0
        return self
        # self.tail = cap-1

    def __new__(cls, cap=16):
        return init(cap)

    BaseProxy.__new__ = __new__
    return init

if __name__ == '__main__':
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    from time import time

    class IntMemoryStruct(nb.types.StructRef): pass
    class IntMemoryProxy(structref.StructRefProxy): pass
    
    IntMemory = TypedMemory(IntMemoryStruct, IntMemoryProxy, np.int32)
    start = time()
    ints = IntMemory(16)
    ints.push(5)
    ints.pop(0)
    print(time()-start)
    aaaa
    
    IntMemory = TypedMemory(np.uint32)
    points = PointMemory(2)
    lst = IntMemory(2)
    
    @nb.njit
    def test(points):
        for i in range(10240000):
            points.push((1,2))
        
    from time import time
    test(points)
    
    points = PointMemory(10240000)
    start = time()
    test(points)
    print(time()-start)
    
    
    '''
    p2 = points.push(np.void((2,1), t_point))
    print('size:', len(points))
    print('get', p1, points[p1])
    print('get', p2, points[p2])
    print('pop', p2, points.pop(p2))
    print('size:', len(points))
    '''
    
