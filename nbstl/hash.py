import numba as nb
import numpy as np

from memory import TypedMemory, sub_class

class Hash: # [rep] Hash->TypedHash
    def __init__(self, cap=16, ktype=np.int32, vtype=None): # [rep] , ktype=np.int32, vtype=None->
        self.cap = cap
        self.size = 0
        # 0:blank, 1:has, 2:removed
        self.msk = np.zeros(cap, dtype=np.uint8)
        self.key = np.zeros(cap, dtype=ktype)
        self.body = np.zeros(cap, dtype=vtype) # [if map]
    
    # insert at 0 or 2
    def push(self, obj, value=None):
        cap = self.cap
        k = 31*hash(obj)%cap
        msk = self.msk
        key = self.key
        for i in range(cap):
            cur = (k+i)%cap
            if msk[cur]!=1: break
            if key[cur]==obj: return False
        # if i>100: print(i, cap, self.size/cap)
        key[cur] = obj
        self.body[cur] = value # [if map][attr] self.body[cur]<-value
        msk[cur] = 1
        self.size += 1
        if self.size > cap*2/3:
            self.expand()
        return True
    
    def has(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        msk = self.msk
        key = self.key
        for i in range(cap):
            cur = (k+i)%cap
            if msk[cur]==0: return False
            if msk[cur]==2: continue
            if key[cur]==obj: return True
        return False

    def pop(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        msk = self.msk
        key = self.key
        
        for i in range(cap):
            cur = (k+i)%cap
            if msk[cur]==0: return
            if msk[cur]==2: continue
            if key[cur]==obj:
                msk[cur] = 2
                self.size -= 1
                return self.body[cur] # [if map]
                return None
        return

    def get(self, obj):
        cap = self.cap
        k = 31*hash(obj)%cap
        msk = self.msk
        key = self.key
        
        for i in range(cap):
            cur = (k+i)%cap
            if msk[cur]==0: return
            if msk[cur]==2: continue
            if key[cur]==obj:
                return self.body[cur] # [if map]
                return None
        return
    
    def expand(self):
        # print('in')
        omsk = self.msk
        okey = self.key
        obody = self.body # [if map]
        cap = self.cap * 2
        self.cap = cap
        self.size = 0
        msk = np.zeros(cap, dtype=np.uint8)
        key = np.zeros(cap, dtype=okey.dtype)
        body = np.zeros(cap, dtype=obody.dtype) # [if map]
        for i in range(cap//2):
            if omsk[i]!=1: continue
            obj = okey[i]
            cap = self.cap
            k = 31*hash(obj)%cap
            for j in range(cap):
                cur = (k+j)%cap
                if msk[cur]!=1: break
            key[cur] = obj
            msk[cur] = 1
            body[cur] = obody[i] # [if map]
            self.size += 1
        self.msk = msk
        self.key = key
        self.body = body # [if map]
        # print(self.cap, 'out')

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val): 
        self.push(key, val)
    
    def __len__(self):
        return self.size
    
    def toarray(self):
        return self.key[self.msk==1]

class MemoryHash:
    def __init__(self, cap=128, memory=None):
        self.hash = IntHash(cap)
        self.memory = memory if memory is not None else typememory(cap)

    def push(self, key, value):
        self.hash.push(key, self.memory.push(value))

    def pop(self, key):
        idx = self.hash.pop(key)
        if idx is None: return
        return self.memory.pop(int(idx))

    def get(self, key):
        idx = self.hash.get(key)
        if idx is None: return
        return self.memory.body[int(idx)]

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val): 
        self.push(key, val)
    
    def __len__(self):
        return self.hash.size 

def type_hash(ktype, vtype=None):
    fields = [('cap', nb.uint32), ('size', nb.uint32),
              ('msk', nb.uint8[:]), ('key', nb.from_dtype(ktype)[:])]
    if vtype: fields.append(('body', nb.from_dtype(vtype)[:]))
    
    local = {'ktype':ktype, 'vtype':vtype, 'np':np}
    typedhashset = sub_class(Hash, vtype, map=vtype is not None)
    # print(typedhashset)
    exec(typedhashset, local)
    TypedHash = local['TypedHash']
    if vtype is None: del TypedHash.get
    return nb.experimental.jitclass(fields)(TypedHash)

def memory_hash(ktype, typememory):
    IntHash = type_hash(ktype, np.int32)
    local = {'typememory':typememory, 'IntHash':IntHash}
    memoryhash = sub_class(MemoryHash, None, map=True)

    exec(memoryhash, local)
    TypedHash = local['MemoryHash']
    fields = [('hash', IntHash.class_type.instance_type),
              ('memory', typememory.class_type.instance_type)]
    return nb.experimental.jitclass(fields)(TypedHash)

def TypedHash(ktype, dtype=None):
    if hasattr(dtype, 'class_type'):
        return memory_hash(ktype, dtype)
    else: return type_hash(ktype, dtype)
    
def print_hash(hs):
    for m,v in zip(hs.msk, hs.key):
        print(m,v)

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])

    PointHash = TypedHash(np.dtype('<8U'), t_point)
    points = PointHash(128)
    points['p1'] = (1,5)
    points['p2'] = (3,3)

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
