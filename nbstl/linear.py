import numpy as np
import numba as nb

from memory import TypedMemory, sub_class

class Deque: # [rep] Deque->TypedDeque
    def __init__(self, cap, dtype=np.uint32): # [rep] , dtype=np.uint32-> 
        self.body = np.zeros(cap, dtype=dtype)
        self.cap = cap
        self.head = 0
        self.tail = cap-1
        self.size = 0

    def push_front(self, obj):
        if self.size==self.cap: self.expand()
        self.size += 1
        self.head = (self.head-1)%self.cap
        self.body[self.head] = obj # [attr] self.body[self.head]<-obj

    def push_back(self, obj):
        if self.size==self.cap: self.expand()
        self.size += 1
        self.tail = (self.tail+1)%self.cap
        self.body[self.tail] = obj # [attr] self.body[self.tail]<-obj

    def pop_front(self):
        self.size -= 1
        rst = self.body[self.head]
        self.head = (self.head+1)%self.cap
        return rst

    def pop_back(self):
        self.size -= 1
        rst = self.body[self.tail]
        self.tail = (self.tail-1)%self.cap
        return rst

    def first(self, idx=0):
        return self.body[(self.head+idx)%self.cap]

    def last(self, idx=0):
        return self.body[self.tail-idx]

    def expand(self):
        self.body = np.concatenate(
            (self.body[self.head:],
             self.body[:(self.tail+1)%self.cap],
             np.zeros(self.cap, self.body.dtype)))
        self.head = 0
        self.tail = self.cap-1
        self.cap *= 2

def type_deque(dtype, mode='deque'):
    fields = [('head', nb.int32), ('tail', nb.int32),
          ('cap', nb.uint32), ('size', nb.uint32),
          ('body', nb.from_dtype(dtype)[:])]

    local = {'Deque':Deque, 'dtype':dtype, 'np':np}
    
    typeddeque = sub_class(Deque, dtype)
    # print(typeddeque)
    exec(typeddeque, local)
    TypedDeque = local['TypedDeque']
    
    if mode=='stack':
        TypedDeque.push = TypedDeque.push_back
        TypedDeque.pop = TypedDeque.pop_back
        TypedDeque.top = TypedDeque.last
    if mode=='queue':
        TypedDeque.push = TypedDeque.push_back
        TypedDeque.pop = TypedDeque.pop_front
        TypedDeque.top = TypedDeque.first
    if mode!='deque':
        del TypedDeque.push_back, TypedDeque.push_front
        del TypedDeque.pop_back, TypedDeque.pop_front
        del TypedDeque.first, TypedDeque.last
    return nb.experimental.jitclass(fields)(TypedDeque)

IntDeque = type_deque(np.uint32)

class MemoryDeque:
    def __init__(self, cap=128, memory=None):
        self.queue = IntDeque(cap)
        self.memory = memory if memory is not None else typememory(cap)

    def first(self, idx=0):
        return self.memory.body[self.queue.first(idx)]

    def last(self, idx=0):
        return self.memory.body[self.queue.last(idx)]

    def push_front(self, obj):
        self.queue.push_front(self.memory.push(obj))

    def push_back(self, obj):
        self.queue.push_back(self.memory.push(obj))

    def pop_front(self):
        return self.memory.pop(self.queue.pop_front())

    def pop_back(self):
        return self.memory.pop(self.queue.pop_back())

    @property
    def size(self): return self.queue.size
    
    def __len__(self):
        return self.queue.size

def memory_deque(typememory, mode='deque'):
    queue_type = nb.deferred_type()
    queue_type.define(IntDeque.class_type.instance_type)
    
    memory_type = nb.deferred_type()
    memory_type.define(typememory.class_type.instance_type)
    
    local = {'typememory': typememory, 'IntDeque': IntDeque}
    memorydeque = sub_class(MemoryDeque, None)
    # print(memorydeque)
    exec(memorydeque, local)
    TypedDeque = local['MemoryDeque']
    
    if mode=='stack':
        TypedDeque.push = TypedDeque.push_back
        TypedDeque.pop = TypedDeque.pop_back
        TypedDeque.top = TypedDeque.last
    if mode=='queue':
        TypedDeque.push = TypedDeque.push_back
        TypedDeque.pop = TypedDeque.pop_front
        TypedDeque.top = TypedDeque.first
    if mode!='deque':
        del TypedDeque.push_back, TypedDeque.push_front
        del TypedDeque.pop_back, TypedDeque.pop_front
        del TypedDeque.first, TypedDeque.last
        
    fields = [('queue', nb.optional(queue_type)),
              ('memory', nb.optional(memory_type))]
    
    return nb.experimental.jitclass(fields)(TypedDeque)

def TypedDeque(dtype):
    if hasattr(dtype, 'class_type'):
        return memory_deque(dtype, 'deque')
    else: return type_deque(dtype, 'deque')

def TypedStack(dtype):
    if hasattr(dtype, 'class_type'):
        return memory_deque(dtype, 'stack')
    else: return type_deque(dtype, 'stack')

def TypedQueue(dtype):
    if hasattr(dtype, 'class_type'):
        return memory_deque(dtype, 'queue')
    else: return type_deque(dtype, 'queue')

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])

    PointMemory = TypedMemory(t_point)
    PointDeque = TypedDeque(PointMemory)


    points = PointMemory(5)
    
    que1 = PointDeque(16, points)
    que2 = PointDeque(16, points)
    
    abcd
    points = PointDeque(1024000)

    
    @nb.njit
    def test(points):
        for i in range(1024000-1):
            points.push_back((1, 2))

    test(points)
    points = PointDeque(1024000)
    start = time()
    test(points)
    print('push 102400 xy points with memory mode:', time()-start)



    PointDeque = TypedDeque(t_point)
    points = PointDeque(1024000)
    
    @nb.njit
    def test(points):
        for i in range(1024000-1):
            points.push_back((1, 2))

    test(points)
    points = PointDeque(1024000)
    start = time()
    test(points)
    print('push 102400 xy points with dtype mode:', time()-start)
    
