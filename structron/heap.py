import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core.extending import overload_method

@nb.njit(cache=True)
def expand(self):
    self.body = np.concatenate((self.body, self.body))
    self.cap *= 2

def build_push(mode='set'):
    @nb.njit(cache=True)
    def push(self, k, v=None):
        if self.size == self.cap:
            self.expand()
        i = self.size
        body = self.body

        if mode=='map': ek = k
        if mode=='set' or mode=='eval': ek = self.eval(k)

        while i != 0:
            pi = (i - 1) // 2
            if mode=='comp':
                if self.comp(body[pi], k) <= 0: break
            else:
                if self.eval(body[pi]) <= ek: break
            
            body[i] = body[pi]
            i = pi

        if mode=='map':
            body[i]['key'], body[i]['value'] = k, v
        else: body[i] = k        
        self.size += 1
    return push

push_set = build_push('set')
push_map = build_push('map')
push_eval = build_push('eval')
push_comp = build_push('comp')

def build_pop(mode='set'):
    @nb.njit(cache=True)
    def pop(self):
        if self.size == 0: return
        self.size -= 1
        size = self.size
        
        body = self.body
        self.buf[0] = body[0]
        last = body[size]
        
        i = 0
        while True:
            ci = 2 * i + 1
            if ci >= size: break

            if mode!='comp':
                br = self.eval(body[ci]) >= self.eval(body[ci + 1])
            else: br = self.comp(body[ci], body[ci+1]) >=0
            if ci + 1 < size and br: ci += 1
            if mode!='comp':
                if self.eval(last) <= self.eval(body[ci]): break
            else:
                if self.comp(last, body[ci]) <= 0: break
                
            body[i] = body[ci]
            i = ci
        
        body[i] = last
        return self.buf[0]
    return pop

pop_comp = build_pop('comp')
pop_set = build_pop('set')

@nb.njit
def top(self): return self.body[0]

@nb.njit
def clear(self): self.size = 0

def istype(obj):
    if isinstance(obj, np.dtype): return True
    return isinstance(obj, type) and isinstance(np.dtype(obj), np.dtype)

def TypedHeap(BaseStruct, BaseProxy, ktype=None, vtype=None, attrs={}):
    import inspect
    if ktype is None:
        mode = 'set'
        ktype = lambda self, x: x
    elif not istype(ktype):
        n = len(inspect.signature(ktype).parameters)
        mode = 'eval' if n==2 else 'comp'
    else:
        vtype = np.dtype([('key', ktype), ('value', vtype)])
        mode, ktype = 'map', lambda self, x: x['key']
    
    struct = structref.register(BaseStruct)

    t_heap = struct([('cap', nb.int32), ('size', nb.int32),
                     ('body', nb.from_dtype(vtype)[:]),
                     ('buf', nb.from_dtype(vtype)[:])] +
                    [(k, nb.from_dtype(v)) for k,v in attrs.items()])
    
    push = {'set': push_set, 'comp': push_comp, 'map': push_map, 'eval': push_eval}[mode]
    pop = {'comp': pop_comp, 'set': pop_set, 'map': pop_set, 'eval': pop_set}[mode]
    
    temp = ''' # add attrs
        def get_%s(self): return self.%s
        def set_%s(self, v): self.%s = v
        BaseProxy.%s = property(nb.njit(get_%s), nb.njit(set_%s))
        del get_%s, set_%s'''
    temp = '\n'.join([i.strip() for i in temp.split('\n')])

    for i in ('size', 'cap', 'key', 'body', 'buf'): exec(temp%((i,)*9))
    for i in attrs: exec(temp%((i,)*9))

    BaseProxy.expand = expand
    BaseProxy.push = push
    BaseProxy.pop = pop
    BaseProxy.top = top
    BaseProxy.clear = clear

    structref.define_boxing(struct, BaseProxy)

    overload_method(struct, "push")(lambda self, k, v=None: push.py_func)
    overload_method(struct, "pop")(lambda self: pop.py_func)
    overload_method(struct, "expand")(lambda self: expand.py_func)
    overload_method(struct, "top")(lambda self: top.py_func)
    overload_method(struct, "clear")(lambda self: clear.py_func)
    # print(ktype)
    if mode!='comp': overload_method(struct, 'eval')(lambda self, x: ktype)
    else: overload_method(struct, 'comp')(lambda self, x1, x2: ktype)
    
    @nb.njit(cache=True)
    def init(cap):
        self = structref.new(t_heap)
        self.cap = cap
        self.size = 0
        self.body = np.empty(cap, dtype=vtype)
        self.buf = np.empty(1, dtype=vtype)
        return self

    def __new__(cls, cap=16):
        return init(cap)
    
    
    BaseProxy.__new__ = __new__
    return init

if __name__ == '__main__':    
    from time import time
    # hist = np.load('hist.npy')
    hist = np.random.rand(1000000)
    
    class FloatHeapStruct(nb.types.StructRef): pass
    class FloatHeapProxy(structref.StructRefProxy): pass
    
    FloatHeap = TypedHeap(FloatHeapStruct, FloatHeapProxy, None, np.float64)
    floats = FloatHeap(16)
    aaaa
    # FloatHeap = TypedHeap(np.float64)
    
    @nb.njit #(cache=True)
    def test(points, hist):
        for i in range(1000000): points.push(np.random.rand())
        for i in range(1000000): points.pop()
        return points
    
    from heapq import heappush, heappop
    
    def heaptest(hist):
        lst = []
        for i in hist: heappush(lst, i)
        for i in hist: heappop(lst)
    
    points = FloatHeap(16)
    start = time()
    r = test(points, hist)
    print(time()-start)
    
    points = FloatHeap(16)
    start = time()
    test(points, hist)
    print(time()-start)

    hist = hist.tolist()
    start = time()
    heaptest(hist)
    print(time()-start)
    
