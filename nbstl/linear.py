import numpy as np
import numba as nb

from .memory import TypedMemory, make_names, make_attrs

class Deque:
    def __init__(self, cap):
        self.body = np.zeros(cap, dtype=np.uint32)
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

    names, attrs = make_names(dtype), make_attrs(dtype, ' '*12)
    local = {'Deque':Deque, 'dtype':dtype, 'np':np}
    
    typeddeque = '''
    class TypedDeque(Deque):
        def __init__(self, cap):
            self.body = np.zeros(cap, dtype)
            self.cap = cap
            self.head = 0
            self.tail = cap-1
            self.size = 0

        def push_front(self, %s):
            if self.size==self.cap: self.expand()
            self.size += 1
            self.head = (self.head-1)%%self.cap
            cur = self.head
            %s

        def push_back(self, %s):
            if self.size==self.cap: self.expand()
            self.size += 1
            self.tail = (self.tail+1)%%self.cap
            cur = self.tail
            %s

    '''%(names, attrs, names, attrs)
    # print(typeddeque)

    typeddeque = '\n'.join([i[4:] for i in typeddeque.split('\n')])
    exec(typeddeque, local)
    TypedDeque = local['TypedDeque']
    if mode=='deque': pass
    if mode=='stack':
        TypedDeque.push = TypedDeque.push_back
        TypedDeque.pop = TypedDeque.pop_back
        TypedDeque.top = TypedDeque.last
    if mode=='queue':
        TypedDeque.push = TypedDeque.push_back
        TypedDeque.pop = TypedDeque.pop_front
        TypedDeque.top = TypedDeque.first
    return nb.experimental.jitclass(fields)(TypedDeque)

IntDeque = type_deque(np.uint32)

def memory_deque(typememory, mode='deque'):
    queue_type = nb.deferred_type()
    queue_type.define(IntDeque.class_type.instance_type)
    
    memory_type = nb.deferred_type()
    memory_type.define(typememory.class_type.instance_type)

    names = typememory.class_type.struct['body'].dtype.dtype.names
    names = ', '.join(names)
    
    local = {'typememory': typememory, 'IntDeque': IntDeque}
    memorydeque = '''
    class MemoryDeque:
        def __init__(self, cap=128, memory=None):
            self.queue = IntDeque(cap)
            self.memory = memory or typememory(cap)

        def first(self):
            return self.memory.body[self.queue.first()]

        def last(self):
            return self.memory.body[self.queue.last()]

        def push_front(self, %s):
            self.queue.push_front(self.memory.push(%s))

        def push_back(self, %s):
            self.queue.push_back(self.memory.push(%s))

        def pop_front(self):
            return self.memory.pop(self.queue.pop_front())

        def pop_back(self):
            return self.memory.pop(self.queue.pop_back())

        @property
        def size(self): return self.queue.size
        
        def __len__(self):
            return self.queue.size
    ''' % (names, names, names, names)
    # print(memorydeque)

    memorydeque = '\n'.join([i[4:] for i in memorydeque.split('\n')])
    exec(memorydeque, local)
    MemoryDeque = local['MemoryDeque']
    if mode=='deque': pass
    if mode=='stack':
        MemoryDeque.push = MemoryDeque.push_back
        MemoryDeque.pop = MemoryDeque.pop_back
        MemoryDeque.top = MemoryDeque.last
    if mode=='queue':
        MemoryDeque.push = MemoryDeque.push_back
        MemoryDeque.pop = MemoryDeque.pop_front
        MemoryDeque.top = MemoryDeque.first
        
    fields = [('queue', nb.optional(queue_type)),
              ('memory', nb.optional(memory_type))]
    
    return nb.experimental.jitclass(fields)(MemoryDeque)

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

    # PointMemory = type_memory(t_point)

    PointMemory = TypedMemory(t_point)
    PointDeque = TypedDeque(PointMemory)
    points = PointDeque(102400)
    
    @nb.njit
    def test(points):
        for i in range(102400-1):
            points.push_back(1, 2)

    test(points)
    start = time()
    test(points)
    print('push 102400 xy points with memory mode:', time()-start)



    PointDeque = TypedDeque(t_point)
    points = PointDeque(102400)
    
    @nb.njit
    def test(points):
        for i in range(102400-1):
            points.push_back(1, 2)

    test(points)
    start = time()
    test(points)
    print('push 102400 xy points with dtype mode:', time()-start)
    
