import numpy as np
import numba as nb

mode = 'deque'
class Deque:
    def __init__(self, dtype, cap): 
        self.body = np.zeros(cap, dtype=dtype)
        self.cap = cap
        self.head = 0
        self.tail = cap-1
        self.size = 0

    def push_front(self, obj):
        if self.size==self.cap: self.expand()
        self.size += 1
        self.head = (self.head-1)%self.cap
        self.body[self.head] = obj

    def push_back(self, obj):
        if self.size==self.cap: self.expand()
        self.size += 1
        self.tail = (self.tail+1)%self.cap
        self.body[self.tail] = obj

    def pop_front(self):
        if self.size==0: return
        self.size -= 1
        rst = self.body[self.head]
        self.head = (self.head+1)%self.cap
        return rst

    def pop_back(self):
        if self.size==0: return
        self.size -= 1
        rst = self.body[self.tail]
        self.tail = (self.tail-1)%self.cap
        return rst

    def first(self, idx=0):
        if self.size==0: return
        return self.body[(self.head+idx)%self.cap]

    def last(self, idx=0):
        if self.size==0: return
        return self.body[self.tail-idx]

    def expand(self):
        self.body = np.concatenate(
            (self.body[self.head:],
             self.body[:(self.tail+1)%self.cap],
             np.zeros(self.cap, self.body.dtype)))
        self.head = 0
        self.tail = self.cap-1
        self.cap *= 2

    def __len__(self): return self.size

def type_deque(dtype, typemode='deque'):
    global mode; mode = typemode
    fields = [('head', nb.int32), ('tail', nb.int32),
          ('cap', nb.uint32), ('size', nb.uint32),
          ('body', nb.from_dtype(dtype)[:])]

    class TypedDeque(Deque):
        _init_ = Deque.__init__
        def __init__(self, cap=16):
            self._init_(dtype, cap)

    if mode=='stack':
        TypedDeque.push = TypedDeque.push_back
        TypedDeque.pop = TypedDeque.pop_back
        TypedDeque.top = TypedDeque.last
    
    if mode=='queue':
        TypedDeque.push = TypedDeque.push_back
        TypedDeque.pop = TypedDeque.pop_front
        TypedDeque.top = TypedDeque.first

    return nb.experimental.jitclass(fields)(TypedDeque)

def TypedDeque(dtype): return type_deque(dtype, 'deque')

def TypedStack(dtype): return type_deque(dtype, 'stack')

def TypedQueue(dtype): return type_deque(dtype, 'queue')

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])

    PointDeque = TypedQueue(t_point)
    
    points = PointDeque()
