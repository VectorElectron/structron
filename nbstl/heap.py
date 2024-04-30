import numpy as np
import numba as nb
from .memory import TypedMemory, make_names, make_attrs

class Heap:
    def __init__(self, cap):
        self.cap = cap
        self.idx = np.zeros(cap, dtype=np.float32)
        self.body = np.zeros(cap, dtype=t_point)
        self.size = 0

    def push(self, k, v):
        if self.size == self.cap: self.expand()
        i = self.size
        idx = self.idx
        idx[i] = k
        self.body[i] = v
        
        # heapsize += 1
        while (i!=0) & (heap[(i-1)//2]>heap[i]):
            self.swap((i-1)//2, i)
            # print(heap, i, '---')
            i = (i-1) // 2
        self.size += 1

    def expand(self):
        self.idx = np.concatenate(
            (self.idx, np.zeros(self.cap, self.idx.dtype)))
        self.body = np.concatenate(
            (self.body, np.zeros(self.cap, self.body.dtype)))
        self.cap *= 2
        
    def swap(self, i1, i2):
        # print('swap', i1, i2)
        idx = self.idx
        body = self.body
        
        idx[i1], idx[i2] = idx[i2], idx[i1]
        body[i1], body[i2] = body[i2], body[i1]


    def pop(self):
        # if self.size == 0: return None
        self.size -= 1
        self.swap(0, self.size)
        self.heapfy(0)
        return self.body[self.size]

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

def make_swap(dtype=None, prefix=''):
    if hasattr(dtype, 'names'):
        namespair = [(i,i,i,i) for i in dtype.names]
        line = 'obj1.%s, obj2.%s = obj2.%s, obj1.%s'
        lines = [line%i for i in namespair]
        lines.insert(0, 'obj1, obj2 = body[i1], body[i2]')
        return ('\n'+prefix).join(lines)
    return 'body[i1], body[i2] = body[i2], body[i1]'
    
def type_heap(dtype, ktype=np.float32):
    fields = [('size', nb.uint32), ('cap', nb.uint32),
              ('idx', nb.from_dtype(ktype)[:]),
              ('body', nb.from_dtype(dtype)[:])]
    
    names, attrs = make_names(dtype), make_attrs(dtype, ' '*12)
    swaps = make_swap(dtype, ' '*12)

    local = {'Heap': Heap, 'ktype':ktype, 'dtype':dtype, 'np':np}
    typedheap = '''
    class TypedHeap(Heap):
        def __init__(self, cap):
            self.cap = cap
            self.idx = np.zeros(cap, dtype=ktype)
            self.body = np.zeros(cap, dtype=dtype)
            self.size = 0
            
        def push(self, k, %s):
            if self.size == self.cap: self.expand()
            i = self.size
            idx = self.idx
            idx[i] = k
            cur = i
            %s
            
            # heapsize += 1
            while (i!=0) & (idx[(i-1)//2]>idx[i]):
                self.swap((i-1)//2, i)
                # print(idx, i, '---')
                i = (i-1) // 2
            self.size += 1

        def swap(self, i1, i2):
            # print('swap', i1, i2)
            idx = self.idx
            body = self.body
            
            idx[i1], idx[i2] = idx[i2], idx[i1]
            %s
    '''%(names, attrs, swaps)
    # print(typedheap)

    typedheap = '\n'.join([i[4:] for i in typedheap.split('\n')])
    exec(typedheap, local)
    TypedHeap = local['TypedHeap']
    return nb.experimental.jitclass(fields)(TypedHeap)

IntHeap = type_heap(np.uint32, np.float32)    


def memory_heap(typememory):
    queue_type = nb.deferred_type()
    queue_type.define(IntHeap.class_type.instance_type)
    
    memory_type = nb.deferred_type()
    memory_type.define(typememory.class_type.instance_type)

    names = typememory.class_type.struct['body'].dtype.dtype.names
    names = ', '.join(names)
    
    local = {'typememory': typememory, 'IntHeap': IntHeap}
    memoryheap = '''
    class MemoryHeap:
        def __init__(self, cap=128, memory=None):
            self.heap = IntHeap(cap)
            self.memory = memory or typememory(cap)

        def push(self, k, %s):
            self.heap.push(k, self.memory.push(%s))

        def pop(self):
            return self.memory.pop(self.heap.pop())
        
        @property
        def size(self): return self.heap.size

        def clear(self): self.heap.clear()
        
        def __len__(self):
            return self.heap.size
    ''' % (names, names)
    # print(memorydeque)

    memoryheap = '\n'.join([i[4:] for i in memoryheap.split('\n')])
    exec(memoryheap, local)
    MemoryHeap = local['MemoryHeap']
    fields = [('heap', nb.optional(queue_type)),
              ('memory', nb.optional(memory_type))]
    
    return nb.experimental.jitclass(fields)(MemoryHeap)

def TypedHeap(dtype):
    if hasattr(dtype, 'class_type'):
        return memory_heap(dtype)
    else: return type_heap(dtype)
    
if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    PointHeap = TypedHeap(t_point)

    points = PointHeap(1024)
    ks = np.random.rand(1024000).astype(np.float32)

    @nb.njit
    def test(points, ks):
        for i in range(1024000):
            points.push(ks[i], 1,1)
        for i in range(1024000):
            points.pop()
            
    test(points, ks)
    points.clear()
    start = time()
    test(points, ks)
    print(time()-start)
    
    PointMemory = TypedMemory(t_point)
    PointHeap = TypedHeap(PointMemory)

    points = PointHeap(1024)
    ks = np.random.rand(1024000).astype(np.float32)

    @nb.njit
    def test(points, ks):
        for i in range(1024000):
            points.push(ks[i], 1,1)
        for i in range(1024000):
            points.pop()
            
    test(points, ks)
    points.clear()
    start = time()
    test(points, ks)
    print(time()-start)
    
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
