import numba as nb
import numpy as np
from numba.experimental import structref
from numba.core.extending import overload_method

def build_push(mode='set'):
    @nb.njit(cache=True)
    def push(self, obj, value=None):
        cap = self.cap
        k = 31*hash(obj)%cap
        # msk = self.msk
        msk = self.idx['msk']
        body = self.idx['body']
        
        for i in range(cap):
            cur = (k+i)%cap
            if msk[cur]!=1: break
            if mode=='set':
                if body[cur]==obj: return
            if mode=='map':
                if body[cur]['key']==obj: return
        
        # if i>100: print(i, cap, self.size/cap)
        if mode == 'set':
            body[cur] = obj
        if mode == 'map':
            body[cur]['key'], body[cur]['value'] = obj, value
        msk[cur] = 1
        self.size += 1
        if self.size > cap*2/3:
            self.expand()
    return push

push_set = build_push('set')
push_map = build_push('map')

def build_has(mode='set'):
    @nb.njit(cache=True)
    def has(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        msk = self.msk
        body = self.body
        for i in range(cap):
            cur = (k+i)%cap
            if msk[cur]==0: return False
            if msk[cur]==2: continue
            if mode=='set':
                if body[cur]==obj: return True
            if mode=='map':
                if body[cur]['key']==obj: return True
        return False
    return has

has_set = build_has('set')
has_map = build_has('map')

def build_pop(mode='set'):
    @nb.njit(cache=True)
    def pop(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        msk = self.idx['msk']
        body = self.idx['body']
        
        for i in range(cap):
            cur = (k+i)%cap
            if msk[cur]==0: return
            if msk[cur]==2: continue
            if mode=='set': key = body[cur]
            if mode=='map': key = body[cur]['key']
            if key==obj:
                msk[cur] = 2
                self.size -= 1
                if mode=='set':
                    return self.body[cur]
                if mode=='map':
                    return self.body[cur]['body']
                return None
        return
    return pop

pop_set = build_pop('set')
pop_map = build_pop('map')

def build_get(mode='set'):
    @nb.njit(cache=True)
    def get(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        msk = self.idx['msk']
        body = self.idx['body']
        
        for i in range(cap):
            cur = (k+i)%cap
            if msk[cur]==0: return
            if msk[cur]==2: continue
            if body[cur]['key']==obj:
                return body[cur]['value']
        return
    return get

get_set = build_get('set')
get_map = build_get('map')

def build_expand(mode='set'):
    @nb.njit(cache=True)
    def expand(self):
        # print('in')
        oidx = self.idx
        omsk, obody = oidx['msk'], oidx['body']
        
        cap = self.cap = self.cap * 2
        self.size = 0
        idx = np.zeros(cap, dtype=oidx.dtype)
        body = idx['body']
        msk = idx['msk']
        
        for i in range(cap//2):
            if omsk[i]!=1: continue
            obj = obody[i]
            if mode=='set': key = obj
            if mode=='map': key = obj['key']
            k = 31*hash(key)%cap
            for j in range(cap):
                cur = (k+j)%cap
                if msk[cur]!=1: break
            body[cur] = obj
            msk[cur] = 1
            self.size += 1
        self.idx = idx
        self.msk = msk
        self.body = body
        # print(self.cap, 'out')
    return expand

expand_set = build_expand(mode='set')
expand_map = build_expand(mode='map')

@nb.njit(cache=True)
def __getitem__(self, key):
    return self.get(key)

@nb.njit(cache=True)
def __setitem__(self, key, val): 
    self.push(key, val)

@nb.njit(cache=True)
def __len__(self):
    return self.size


def TypedHash(BaseStruct, BaseProxy, ktype, vtype=None):
    if ktype is None: mode = 'set'
    else:
        vtype = np.dtype([('key', ktype), ('value', vtype)])
        mode = 'map'

    struct = structref.register(BaseStruct)

    idxtype = np.dtype([('msk', np.uint8), ('body', vtype)])
    
    t_hash = struct([('idx', nb.from_dtype(idxtype)[:]),
                     ('msk', nb.uint8[:]), ('body', nb.from_dtype(vtype)[:]),
                     ('cap', nb.uint32), ('size', nb.uint32)])
    
    push = {'set': push_set, 'map': push_map}[mode]
    pop = {'set': pop_set, 'map': pop_map}[mode]
    has = {'set': has_set, 'map': has_map}[mode]
    expand = {'set': expand_set, 'map': expand_map}[mode]
    get = {'set': get_set, 'map': get_map}[mode]
    
    temp = ''' # add attrs
        def get_%s(self): return self.%s
        def set_%s(self, v): self.%s = v
        BaseProxy.%s = property(nb.njit(get_%s), nb.njit(set_%s))
        del get_%s, set_%s'''
    temp = '\n'.join([i.strip() for i in temp.split('\n')])

    for i in ('msk', 'body', 'cap', 'size'): exec(temp%((i,)*9))

    BaseProxy.expand = expand
    BaseProxy.push = push
    BaseProxy.pop = pop
    BaseProxy.has = has
    BaseProxy.get = get
    BaseProxy.__getitem__ = __getitem__
    BaseProxy.__setitem__ = __setitem__
    BaseProxy.__len__ = __len__

    structref.define_boxing(struct, BaseProxy)

    overload_method(struct, "push")(lambda self, obj, value=None: push.py_func)
    overload_method(struct, "pop")(lambda self: pop.py_func)
    overload_method(struct, "expand")(lambda self: expand.py_func)
    overload_method(struct, "get")(lambda self, obj: get.py_func)
    overload_method(struct, "has")(lambda self, obj: has.py_func)

    @nb.njit(cache=True)
    def init(cap=16):
        self = structref.new(t_hash)
        self.cap = cap
        self.size = 0
        # 0:blank, 1:has, 2:removed
        # self.msk = np.zeros(cap, dtype=np.uint8)
        self.idx = np.zeros(cap, dtype=idxtype)
        self.msk = self.idx['msk']
        self.body = self.idx['body']
        return self

    def __new__(cls, cap=16):
        return init(cap)
    
    BaseProxy.__new__ = __new__
    return init
    
def print_hash(hs):
    for m,v in zip(hs.msk, hs.key):
        print(m,v)

if __name__ == '__main__':
    class IntHashStruct(nb.types.StructRef): pass
    class IntHashProxy(structref.StructRefProxy): pass
    IntHash = TypedHash(IntHashStruct, IntHashProxy, np.int32, np.int32)
    
    ints = IntHash(4)
    
    abcd
    
    '''
    StrMemory = TypedMemory(np.dtype('<8U'))

    StrHashMap = TypedHash(np.int32, StrMemory)
    strings = StrMemory()
    # hs = StrHashMap(128)

    
    numbs = np.random.randint(0,10000000, 10000000)    
    # numbs = np.array([1,3,5,7,9])
    
    @nb.njit
    def nbsettest():
        s = {}
        for i in numbs:
            s[i] = 'abc'
        return s
    
    @nb.njit
    def mysettest(hs):
        for i in numbs:
            hs.push(i, 'abc')
        return hs
    
    from time import time
    nbsettest()
    start = time()
    s = nbsettest() # 0.8
    print(time()-start)

    hs = StrHashMap(128, strings)
    mysettest(hs)
    start = time()
    hs = StrHashMap(128, strings)
    hs = mysettest(hs) # 0.8
    print(time()-start)
    
    '''
