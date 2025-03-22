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

        if mode=='eval': body = self.body
        if mode=='comp': body = self.body
        if mode=='set': key = self.key
        if mode=='map': key, body = self.key, self.body

        while i!=0:
            pi = (i-1)//2
            if mode=='set': br = key[pi] - k
            if mode=='map': br = key[pi] - k
            if mode=='eval': br = self.eval(body[pi])-self.eval(k)
            if mode=='comp': br = self.comp(body[pi], k)

            if br<=0: break

            if mode=='set': key[i] = key[pi]
            if mode=='map': key[i] = key[pi]
            if mode!='set': body[i] = body[pi]
            i = pi

        if mode=='eval': body[i] = k
        if mode=='comp': body[i] = k
        if mode=='set': key[i] = k
        if mode=='map': key[i], body[i] = k, v
            
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

    def pop(self):
        if self.size == 0: return
        self.size -= 1
        # self.swap(0, self.size)
        size = self.size
        
        if mode=='set':
            key = self.key
            key[0], key[size] = key[size], key[0]
            last = key[0]
        if mode=='map':
            key = self.key
            body = self.body
            self.buf[0] = body[0]
            last = key[size]
        if mode=='eval':
            body = self.body
            self.buf[0] = body[0]
            last = body[size]
        if mode=='comp':
            body = self.body
            self.buf[0] = body[0]
            last = body[size]
        
        i = 0
        while True:
            ci = 2 * i + 1
            if ci>=size: break
            
            if mode=='set':
                if ci+1<size and key[ci]>=key[ci+1]: ci+=1
                if last <= key[ci]: break
            if mode=='map':
                if ci+1<size and key[ci]>=key[ci+1]: ci+=1
                if last <= key[ci]: break
            if mode=='eval':
                if ci+1<size and self.eval(body[ci])>=self.eval(body[ci+1]): ci += 1
                if self.eval(last)<=self.eval(body[ci]): break
            if mode=='comp':
                if ci+1<size and self.comp(body[ci], body[ci+1])>=0: ci += 1
                if self.comp(last, body[ci])<=0: cbreak
                
            if mode=='set': key[i] = key[ci]
            if mode=='map': key[i] = key[ci]
            if mode!='set': body[i] = body[ci]
            i = ci
            
        if mode=='set': key[i] = last
        if mode=='map': key[i], body[i] = last, body[size]
        if mode=='eval': body[i] = last
        if mode=='comp': body[i] = last
                
        if mode!='set': return self.buf[0]
        return key[size]

    def top(self):
        if mode!='set':
            return self.body[0]
        return self.key[0]

    def topkey(self): return self.key[0]

    def topvalue(self): return self.body[0]

    def clear(self): self.size = 0

    def __setitem__(self, key, val): 
        self.push(key, val)
    
    def __len__(self):
        return self.size

def istype(obj):
    if isinstance(obj, np.dtype): return True
    return isinstance(obj, type) and isinstance(np.dtype(obj), np.dtype)
                 
def TypedHeap(ktype, vtype=None, jit=True):
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
    if not jit: return TypedHeap
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
    
    FloatHeap = TypedHeap(np.float64, jit=True)
    ints = FloatHeap(16)
    
    # hist = np.load('hist.npy')
    hist = np.random.rand(1000000)
    
    @nb.njit
    def test(hist):
        points = FloatHeap(128)
        for i in range(1000000): points.push(np.random.rand())
        for i in range(1000000): points.pop()

    from heapq import heappush, heappop
    
    def heaptest(hist):
        lst = []
        for i in hist: heappush(lst, i)
        for i in hist: heappop(lst)


    test(hist)

    start = time()
    test(hist)
    print(time()-start)

    hist = hist.tolist()
    start = time()
    heaptest(hist)
    print(time()-start)
