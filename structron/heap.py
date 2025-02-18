import numpy as np
import numba as nb

mode = 'set'

class Heap:
    def __init__(self, ktype, dtype, cap=16):
        self.cap = cap
        if mode!='func':
            self.key = np.zeros(cap, dtype=ktype)
        if mode!='set':
            self.body = np.zeros(cap, dtype=dtype)
            self.buf = np.zeros(1, dtype=dtype)
        self.size = 0
    
    def push(self, k, v=None):
        if self.size == self.cap: self.expand()
        i = self.size

        if mode != 'func':
            key = self.key
            key[i] = k
        if mode!='set':
            body = self.body
            body[i] = v

        while i!=0:
            if mode!='func':
                if key[(i-1)//2]<=key[i]: break
            if mode=='func':
                if self.eval(body[(i-1)//2])<=self.eval(body[i]): break
            self.swap((i-1)//2, i)
            i = (i-1) // 2
        self.size += 1

    def expand(self):
        if mode!='func':
            self.key = np.concatenate(
                (self.key, np.zeros(self.cap, self.key.dtype)))
        if mode!='set':
            self.body = np.concatenate((self.body, self.body))
        self.cap *= 2
        
    def swap(self, i1, i2):
        if mode!='func':
            key = self.key
            key[i1], key[i2] = key[i2], key[i1]
        if mode!='set':
            body = self.body
            self.buf[0] = body[i1]
            body[i1] = body[i2]
            body[i2] = self.buf[0]

    def pop(self):
        if self.size == 0: return
        self.size -= 1
        self.swap(0, self.size)
        self.heapfy(0)
        if mode!='set':
            return self.body[self.size]
        return self.key[self.size]

    def top(self):
        if mode!='set':
            return self.body[0]
        return self.key[0]
    
    def heapfy(self, i):
        if mode!='func': key = self.key
        if mode=='func': body = self.body
        
        size = self.size
        while True:
            l = 2 * i + 1 
            r = 2 * i + 2
            
            smallest = i
            if mode!='func':
                if (l < size) and (key[l] < key[smallest]): smallest = l; 
                if (r < size) and (key[r] < key[smallest]): smallest = r;
            if mode=='func':
                if (l < size) and (self.eval(body[l]) < self.eval(body[smallest])):
                    smallest = l; 
                if (r < size) and (self.eval(body[r]) < self.eval(body[smallest])):
                    smallest = r;
            if smallest == i: break
            else:
                self.swap(i, smallest)
                i = smallest

    def clear(self): self.size = 0

    def __setitem__(self, key, val): 
        self.push(key, val)
    
    def __len__(self):
        return self.size

def istype(obj):
    if isinstance(obj, np.dtype): return True
    return isinstance(obj, type) and isinstance(np.dtype(obj), np.dtype)

def TypedHeap(ktype, vtype=None):
    global mode
    if not istype(ktype): mode = 'func'
    elif vtype is None: mode = 'set'
    else: mode = 'map'
    
    fields = [('size', nb.uint32), ('cap', nb.uint32)]
    if mode in {'set', 'map'}:
        fields.append(('key', nb.from_dtype(ktype)[:]))
    if mode in {'map', 'func'}:
        fields += [
              ('body', nb.from_dtype(vtype)[:]),
              ('buf', nb.from_dtype(vtype)[:])]
    
    class TypedHeap(Heap):
        _init_ = Heap.__init__
        if mode=='func': eval = ktype
        def __init__(self, cap):
            self._init_(None if mode=='func' else ktype, vtype, cap)

    return nb.experimental.jitclass(fields)(TypedHeap)  

    
if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    
    PointHeap = TypedHeap(lambda self, x:x.x, t_point)
    points = PointHeap(128)
    
    points.push(None, np.void((5,5), dtype=t_point))
    points.push(None, np.void((6,6), dtype=t_point))
    points.push(None, np.void((4,4), dtype=t_point))
    
    abcd
    FloatHeap = TypedHeap(np.float32)
    heap = FloatHeap(cap=128)
    

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
