import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core.extending import overload_method

@nb.njit(cache=True)
def push_front(self, obj):
    if self.size==self.cap: self.expand()
    self.size += 1
    self.head = (self.head-1)%self.cap
    self.body[self.head] = obj

@nb.njit(cache=True)
def push_back(self, obj):
    if self.size==self.cap: self.expand()
    self.size += 1
    self.tail = (self.tail+1)%self.cap
    self.body[self.tail] = obj

@nb.njit(cache=True)
def pop_front(self):
    if self.size==0: return
    self.size -= 1
    rst = self.body[self.head]
    self.head = (self.head+1)%self.cap
    return rst

@nb.njit(cache=True)
def pop_back(self):
    if self.size==0: return
    self.size -= 1
    rst = self.body[self.tail]
    self.tail = (self.tail-1)%self.cap
    return rst

@nb.njit(cache=True)
def first(self, idx=0):
    if self.size==0: return
    return self.body[(self.head+idx)%self.cap]

@nb.njit(cache=True)
def last(self, idx=0):
    if self.size==0: return
    return self.body[self.tail-idx]

@nb.njit(cache=True)
def expand(self):
    self.body = np.concatenate(
        (self.body[self.head:],
         self.body[:(self.tail+1)%self.cap],
         np.zeros(self.cap, self.body.dtype)))
    self.head = 0
    self.tail = self.cap-1
    self.cap *= 2

@nb.njit(cache=True)
def __len__(self): return self.size

def TypedLinear(BaseStruct, BaseProxy, dtype, mode='deque'):
    struct = structref.register(BaseStruct)

    t_linear = struct([('head', nb.int32), ('tail', nb.int32),
          ('cap', nb.uint32), ('size', nb.uint32),
          ('body', nb.from_dtype(dtype)[:])])

    temp = ''' # add attrs
        def get_%s(self): return self.%s
        def set_%s(self, v): self.%s = v
        BaseProxy.%s = property(nb.njit(get_%s), nb.njit(set_%s))
        del get_%s, set_%s'''
    temp = '\n'.join([i.strip() for i in temp.split('\n')])

    for i in ('head', 'tail', 'cap', 'size', 'body'): exec(temp%((i,)*9))

    if mode=='deque':
        BaseProxy.push_back = push_back
        BaseProxy.push_front = push_front
        BaseProxy.pop_back = pop_back
        BaseProxy.pop_front = pop_front
        BaseProxy.first = frist
        BaseProxy.last = last

    if mode=='stack':
        BaseProxy.push = push_back
        BaseProxy.pop = pop_back
        BaseProxy.top = last
    
    if mode=='queue':
        BaseProxy.push = push_back
        BaseProxy.pop = pop_front
        BaseProxy.top = first

    structref.define_boxing(struct, BaseProxy)

    overload_method(struct, "expand")(lambda self: expand.py_func)
    if mode=='deque':
        overload_method(struct, "push_back")(lambda self, obj: push_back.py_func)
        overload_method(struct, "push_front")(lambda self, obj: push_front.py_func)
        overload_method(struct, "pop_back")(lambda self: pop_back.py_func)
        overload_method(struct, "pop_front")(lambda self: pop_front.py_func)
        overload_method(struct, "first")(lambda self, idx=0: first.py_func)
        overload_method(struct, "last")(lambda self, idx=0: last.py_func)
    if mode=='stack':
        overload_method(struct, "push")(lambda self, obj: push_back.py_func)
        overload_method(struct, "pop")(lambda self: pop_back.py_func)
        overload_method(struct, "top")(lambda self, idx=0: last.py_func)
    if mode=='queue':
        overload_method(struct, "push")(lambda self, obj: push_back.py_func)
        overload_method(struct, "pop")(lambda self: pop_front.py_func)
        overload_method(struct, "top")(lambda self, idx=0: first.py_func)    
    
    @nb.njit(cache=True)
    def init(cap=16):
        self = structref.new(t_linear)
        self.body = np.zeros(cap, dtype=dtype)
        self.cap = cap
        self.head = 0
        self.tail = cap-1
        self.size = 0
        return self

    def __new__(cls, cap=16):
        return init(cap)

    BaseProxy.__new__ = __new__
    return init

def TypedDeque(BaseStruct, BaseProxy, dtype):
    return TypedLinear(BaseStruct, BaseProxy, dtype, mode='deque')

def TypedStack(BaseStruct, BaseProxy, dtype):
    return TypedLinear(BaseStruct, BaseProxy, dtype, mode='stack')

def TypedQueue(BaseStruct, BaseProxy, dtype):
    return TypedLinear(BaseStruct, BaseProxy, dtype, mode='queue')

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    class FloatHeapStruct(nb.types.StructRef): pass
    class FloatHeapProxy(structref.StructRefProxy): pass
    PointDeque = TypedQueue(FloatHeapStruct, FloatHeapProxy, t_point)
    points = PointDeque()
