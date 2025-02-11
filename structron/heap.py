import numpy as np
import numba as nb
from .memory import TypedMemory, sub_class

class Heap: # [rep] Heap->TypedHeap
    def __init__(self, cap=128, ktype=np.uint32, dtype=np.uint32): # [rep] , ktype=np.uint32, dtype=np.uint32->
        self.cap = cap
        self.idx = np.zeros(cap, dtype=ktype)
        self.body = np.zeros(cap, dtype=dtype) # [if map]
        self.size = 0

    def push(self, key, val=None):
        if self.size == self.cap: self.expand()
        i = self.size
        idx = self.idx
        idx[i] = key
        self.body[i] = val # [if map][attr] self.body[i]<-val
        
        # heapsize += 1
        while (i!=0) & (idx[(i-1)//2]>idx[i]):
            self.swap((i-1)//2, i)
            # print(heap, i, '---')
            i = (i-1) // 2
        self.size += 1

    def expand(self):
        self.idx = np.concatenate(
            (self.idx, np.zeros(self.cap, self.idx.dtype)))
        self.body = np.concatenate((self.body, self.body)) # [if map]
        self.cap *= 2
        
    def swap(self, i1, i2):
        idx = self.idx
        body = self.body # [if map]
        
        idx[i1], idx[i2] = idx[i2], idx[i1]
        body[i1], body[i2] = body[i2], body[i1] # [if map][swap] body[i1]<->body[i2]

    def pop(self):
        if self.size == 0: return
        self.size -= 1
        self.swap(0, self.size)
        self.heapfy(0)
        return self.body[self.size] # [if map]
        return self.idx[self.size]

    def top(self):
        return self.body[0] # [if map]
        return self.idx[0]
    
    def heapfy(self, i):
        idx = self.idx
        size = self.size
        while True:
            l = 2 * i + 1 
            r = 2 * i + 2
            smallest = i
            if (l < size) and (idx[l] < idx[smallest]):
                smallest = l; 
            if (r < size) and (idx[r] < idx[smallest]):
                smallest = r;
            if smallest == i: break
            else:
                self.swap(i, smallest)
                i = smallest

    def clear(self): self.size = 0
        
    def __len__(self): return self.size

def type_heap(ktype, vtype=None):
    fields = [('size', nb.uint32), ('cap', nb.uint32),
              ('idx', nb.from_dtype(ktype)[:])]
    if vtype: fields.append(('body', nb.from_dtype(vtype)[:]))

    local = {'Heap': Heap, 'ktype':ktype, 'dtype':vtype, 'np':np}
    typedheap = sub_class(Heap, vtype, map=vtype is not None)
    # print(typedheap)
    exec(typedheap, local)
    TypedHeap = local['TypedHeap']
    return nb.experimental.jitclass(fields)(TypedHeap)  

class MemoryHeap:
    def __init__(self, cap=128, memory=None):
        self.heap = IntHeap(cap)
        self.memory = memory if memory is not None else typememory(cap)

    def push(self, key, val):
        self.heap.push(key, self.memory.push(val))

    def pop(self):
        if self.heap.size==0: return
        return self.memory.pop(int(self.heap.pop()))

    def top(self):
        return self.memory.body[self.heap.top()]
    
    @property
    def size(self): return self.heap.size

    def clear(self):
        for i in range(self.heap.size):
            self.memory.pop(self.heap.body[i])
        self.heap.clear()
    
    def __len__(self):
        return self.heap.size

def memory_heap(ktype, typememory):
    IntHeap = type_heap(ktype, np.int32)
    queue_type = nb.deferred_type()
    queue_type.define(IntHeap.class_type.instance_type)
    
    memory_type = nb.deferred_type()
    memory_type.define(typememory.class_type.instance_type)

    local = {'typememory': typememory, 'IntHeap': IntHeap}
    memoryheap = sub_class(MemoryHeap, None, map=True)
    # print(memorydeque)

    exec(memoryheap, local)
    TypedHeap = local['MemoryHeap']
    fields = [('heap', nb.optional(queue_type)),
              ('memory', nb.optional(memory_type))]
    
    return nb.experimental.jitclass(fields)(TypedHeap)

def TypedHeap(ktype, dtype=None):
    if hasattr(dtype, 'class_type'):
        return memory_heap(ktype, dtype)
    else: return type_heap(ktype, dtype)
    
if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    
    '''
    FloatHeap = TypedHeap(np.float32, t_point)
    heap = FloatHeap(cap=128)
    abcd

    ks = np.random.rand(1024000).astype(np.float32)

    @nb.njit
    def test(points, ks):
        for i in range(1024000):
            points.push(ks[i])
        for i in range(1024000):
            points.pop()

    points = FloatHeap(1024000+1)
    test(points, ks)
    points.clear()
    start = time()
    test(points, ks)
    print(time()-start)

    abcd
    '''
    
    PointMemory = TypedMemory(t_point)
    points = PointMemory()
    
    PointHeap = TypedHeap(np.float32, PointMemory)
    ph = PointHeap(128, points)
    
    '''
    points = PointHeap(1024001)
    ks = np.random.rand(1024000).astype(np.float32)
    
    @nb.njit
    def test(points, ks):
        for i in range(1024000):
            points.push(ks[i], (1,2))
        for i in range(1024000):
            points.pop()
            
    test(points, ks)
    points.clear()
    start = time()
    test(points, ks)
    print(time()-start)
    '''
    
    '''
    test_data = np.random.randint(0, 100, 100000)
    heap = Heap(len(test_data))

    @njit
    def test(heap, test_data):
        for i in test_data:
            heap.add(i)
        for i in range(len(test_data)):
             heap.pop()

    from time import time

    #test(heap, test_data)
    #heap.size = 0

    start = time()
    #test(heap, test_data)
    print(time()-start)
    '''
