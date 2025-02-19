import numba as nb
import numpy as np

mode = 'set'

class Hash:
    def __init__(self, ktype=np.int32, vtype=None, cap=16):
        self.cap = cap
        self.size = 0
        # 0:blank, 1:has, 2:removed
        # self.msk = np.zeros(cap, dtype=np.uint8)
        self.idx = np.zeros(cap, dtype=ktype)
        if mode=='map':
            self.body = np.zeros(cap, dtype=vtype)
    
    # insert at 0 or 2
    def push(self, obj, value=None):
        cap = self.cap
        k = 31*hash(obj)%cap
        # msk = self.msk
        idx = self.idx
        for i in range(cap):
            cur = (k+i)%cap
            if idx[cur].msk!=1: break
            if idx[cur].key==obj: return
        
        # if i>100: print(i, cap, self.size/cap)
        idx[cur].key = obj
        if mode == 'map':
            self.body[cur] = value
        idx[cur].msk = 1
        self.size += 1
        if self.size > cap*2/3:
            self.expand()
    
    def has(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        idx = self.idx
        for i in range(cap):
            cur = (k+i)%cap
            if idx[cur].msk==0: return False
            if idx[cur].msk==2: continue
            if idx[cur].key==obj: return True
        return False

    def pop(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        idx = self.idx
        
        for i in range(cap):
            cur = (k+i)%cap
            if idx[cur].msk==0: return
            if idx[cur].msk==2: continue
            if idx[cur].key==obj:
                idx[cur].msk = 2
                self.size -= 1
                if mode=='map':
                    return self.body[cur]
                return None
        return

    def get(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        idx = self.idx
        
        for i in range(cap):
            cur = (k+i)%cap
            if idx[cur].msk==0: return
            if idx[cur].msk==2: continue
            if idx[cur].key==obj:
                if mode=='map':
                    return self.body[cur]
                return
        return
    
    def expand(self):
        # print('in')
        oidx = self.idx
        if mode=='map':
            obody = self.body
        cap = self.cap * 2
        self.cap = cap
        self.size = 0
        idx = np.zeros(cap, dtype=oidx.dtype)
        if mode=='map':
            body = np.zeros(cap, dtype=obody.dtype)
        for i in range(cap//2):
            if oidx[i].msk!=1: continue
            obj = oidx[i].key
            cap = self.cap
            k = 31*hash(obj)%cap
            for j in range(cap):
                cur = (k+j)%cap
                if idx[cur].msk!=1: break
            idx[cur].key = obj
            idx[cur].msk = 1
            if mode=='map': body[cur] = obody[i]
            self.size += 1
        self.idx = idx
        if mode=='map': self.body = body
        # print(self.cap, 'out')

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val): 
        self.push(key, val)
    
    def __len__(self):
        return self.size

    def toarray(self):
        return self.idx.key[self.idx.msk==1]

def TypedHash(ktype, vtype=None):
    import inspect
    global mode
    mode = 'set' if vtype is None else 'map'

    key = np.dtype([('msk', np.uint8), ('key', ktype)])
    fields = [('cap', nb.uint32), ('size', nb.uint32),
              ('idx', nb.from_dtype(key)[:])]
    
    if vtype: fields.append(('body', nb.from_dtype(vtype)[:]))

    exec(inspect.getsource(Hash), dict(globals()), locals())
    
    class TypedHash(locals()['Hash']):
        _init_ = Hash.__init__
        def __init__(self, cap):
            self._init_(key, vtype, cap)
    
    return nb.experimental.jitclass(fields)(TypedHash)
    
def print_hash(hs):
    for m,v in zip(hs.msk, hs.key):
        print(m,v)

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])

    PointHash = TypedHash(np.float32, t_point)
    points = PointHash(4)
    points[1] = np.void((1,1), t_point)
    points[2] = np.void((2,2), t_point)
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
