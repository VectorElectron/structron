import numpy as np
import numba as nb

mode = 'set'

class Heap:
    def __init__(self, ktype, dtype, cap=16):
        self.cap = cap
        if mode=='set':
            self.key = np.zeros(cap, dtype=ktype)
        if mode=='map':
            self.key = np.zeros(cap, dtype=ktype)
        if mode!='set':
            self.body = np.zeros(cap, dtype=dtype)
            self.buf = np.zeros(1, dtype=dtype)
        self.size = 0
    
    def push(self, k, v=None):
        if self.size == self.cap: self.expand()
        i = self.size

        if mode=='eval':
            body = self.body
            body[i] = k
        if mode=='comp':
            body = self.body
            body[i] = k
        if mode=='set':
            key = self.key
            key[i] = k
        if mode=='map':
            key = self.key
            body = self.body
            key[i] = k
            body[i] = v

        while i!=0:
            if mode=='set':
                br = key[(i-1)//2] - key[i]
            if mode=='map':
                br = key[(i-1)//2] - key[i]
            if mode=='eval':
                br = self.eval(body[(i-1)//2])-self.eval(body[i])
            if mode=='comp':
                br = self.comp(body[(i-1)//2], body[i])

            if br<=0: break
            self.swap((i-1)//2, i)
            i = (i-1) // 2
        self.size += 1

    def expand(self):
        if mode=='set':
            self.key = np.concatenate(
                (self.key, np.zeros(self.cap, self.key.dtype)))
        if mode=='map':
            self.key = np.concatenate(
                (self.key, np.zeros(self.cap, self.key.dtype)))
        if mode!='set':
            self.body = np.concatenate((self.body, self.body))
        self.cap *= 2
        
    def swap(self, i1, i2):
        if mode=='set':
            key = self.key
            key[i1], key[i2] = key[i2], key[i1]
        if mode=='map':
            key = self.key
            key[i1], key[i2] = key[i2], key[i1]
        if mode!='set':
            body = self.body
            self.buf[0] = body[i1]
            body[i1] = body[i2]
            body[i2] = self.buf[0]

    def pop(self):
        if self.size == 0: return self.body[0]
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
        if mode=='set': key = self.key
        if mode=='map': key = self.key
        if mode!='set': body = self.body
        
        size = self.size
        while True:
            l = 2 * i + 1 
            r = 2 * i + 2
            
            smallest = i

            if mode=='set':
                if (l < size) and key[l]<key[smallest]: smallest = l; 
                if (r < size) and key[r]<key[smallest]: smallest = r;
            if mode=='map':
                if (l < size) and key[l]<key[smallest]: smallest = l; 
                if (r < size) and key[r]<key[smallest]: smallest = r;
                
            if mode=='eval':
                if (l < size) and self.eval(body[l])<self.eval(body[smallest]): smallest = l; 
                if (r < size) and self.eval(body[r])<self.eval(body[smallest]): smallest = r;
            if mode=='comp':
                if (l < size) and self.comp(body[l], body[smallest])<0: smallest = l; 
                if (r < size) and self.comp(body[r], body[smallest])<0: smallest = r;
            
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
    import inspect
    global mode
    if not istype(ktype):
        n = len(inspect.signature(ktype).parameters)
        mode = 'eval' if n==2 else 'comp'
    elif vtype is None: mode = 'set'
    else: mode = 'map'

    exec(inspect.getsource(Heap), dict(globals()), locals())

    fields = [('size', nb.uint32), ('cap', nb.uint32)]
    if mode in {'set', 'map'}:
        fields.append(('key', nb.from_dtype(ktype)[:]))
    if mode in {'map', 'eval', 'comp'}:
        fields += [
              ('body', nb.from_dtype(vtype)[:]),
              ('buf', nb.from_dtype(vtype)[:])]

    class TypedHeap(locals()['Heap']):
        _init_ = Heap.__init__
        if mode=='eval': eval = ktype
        if mode=='comp': comp = ktype

        def __init__(self, cap):
            self._init_(None if mode=='eval' or mode=='comp' else ktype, vtype, cap)
    
    return nb.experimental.jitclass(fields)(TypedHeap)

def print_heap(arr):
    def print_tree(index, level):
        if index < len(arr):
            print_tree(2 * index + 2, level + 1)  # 先打印右子树
            print('    ' * level + str(arr[index]))  # 打印当前节点
            print_tree(2 * index + 1, level + 1)  # 再打印左子树
    print_tree(0, 0)
    
if __name__ == '__main__':
    from time import time
    def f(self, x, y): return x - y
    IntHeap = TypedHeap(f, np.float32)
    ints = IntHeap(128)
    
    x = np.arange(20)
    np.random.shuffle(x)

    for i in x: ints.push(i)


    aaaa
    IntHeap = TypedHeap(np.int32)
    ints = IntHeap(128)
    
    
    abcd
    
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
