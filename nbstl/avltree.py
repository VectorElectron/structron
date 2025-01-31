import numpy as np
import numba as nb
from memory import sub_class, TypedMemory
import inspect

class AVLTree: # [rep] AVLTree->TypedAVLTree
    def __init__(self, cap=128, ktype=np.uint32, dtype=np.uint32): # [rep] , ktype=np.uint32, dtype=np.uint32->
        self.idx = np.zeros(cap, dtype=ilr)
        # self.idx['id'] = np.arange(1, cap+1, dtype=np.int32)
        for i in range(cap): self.idx[i].id = i+1
        self.root = -1
        self.cur = 0
        self.cap = cap
        self.size = 0
        self.tail = cap-1
        # self.key = np.zeros(cap, ktype)
        self.body = np.zeros(cap, dtype=dtype)
        self.hist = np.zeros(256, dtype=np.uint32)

    def expand(self):
        idx = np.zeros(self.cap*2, dtype=ilr)
        for i in range(self.cap*2): idx[i].id = i+1
        self.body = np.concatenate((self.body, self.body))
        idx[:self.cap] = self.idx
        self.idx = idx
        
        self.idx[self.tail].id = self.cap
        self.cur = self.cap
        self.cap *= 2
        self.tail = self.cap - 1

    def get(self, key):
        cur = self.root
        while cur != -1:
            node = self.idx[cur]
            ck = node.key
            if key == ck:
                return self.body[cur]
            if key < ck: cur = node.left
            if key > ck: cur = node.right
    
    def left(self, key):
        cur = parent = self.root
        msk = 0xffffffff>>1
        
        hist = self.hist
        n = 0
        idx = self.idx
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            ck = ilrk.key
            if key == ck: break
            if key < ck: # left
                hist[n] = cur
                cur = ilrk.left
            if key > ck: # right
                hist[n] = cur| 1<<31
                cur = ilrk.right
            n += 1
        node = idx[cur]
        if node.left != -1:
            nxt = node.left
            while True:
                lnode = idx[nxt]
                if lnode.right==-1:
                    return self.body[nxt]
                nxt = lnode.right
        else:
            for i in range(n-1, -1, -1):
                if hist[i]>>31:
                    return self.body[hist[i]&msk]

    def right(self, key):
        cur = parent = self.root
        msk = 0xffffffff>>1
        
        hist = self.hist
        n = 0
        idx = self.idx
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            ck = ilrk.key
            if key == ck: break
            if key < ck: # left
                hist[n] = cur
                cur = ilrk.left
            if key > ck: # right
                hist[n] = cur| 1<<31
                cur = ilrk.right
            n += 1
        node = idx[cur]
        if node.right != -1:
            nxt = node.right
            while True:
                lnode = idx[nxt]
                if lnode.left==-1:
                    return self.body[nxt]
                nxt = lnode.left
        else:
            for i in range(n-1, -1, -1):
                if not hist[i]>>31:
                    return self.body[hist[i]]
    
    def push(self, key, val):
        cur = parent = self.root
        msk = 0xffffffff>>1
        if cur==-1:
            self.root = self.alloc(key, val)
            return
        hist = self.hist
        n = 0
        # find key and record key|(l_r<<31) in history
        idx = self.idx
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            ck = ilrk.key            
            if key == ck: 
                self.body[cur] = val # [attr] self.body[cur]<-val
                return
            hist[n] = cur
            if key < ck: # left
                cur = ilrk.left
            if key > ck: # right
                hist[n] |= 1<<31
                cur = ilrk.right # next
            n += 1

        # add new leaf
        node = self.alloc(key, val)
        idx = self.idx
        ilrk = idx[parent]
        if key<ilrk.key: ilrk.left = node
        else: ilrk.right = node
        
        hist[n] = node
                
        for i in range(n-1, -1, -1):
            # d = dirs[i]
            n = hist[i]
            d = -1 if (n>>31)==0 else 1
            n &= msk
            
            idx[n].bal +=  d
            bal = idx[n].bal
            if bal==0: break
            if bal==2 or bal==-2:
                self.rotate(-1 if i==0 else hist[i-1]&msk,
                    -1 if (hist[i-1]>>31)==0 else 1, hist[i]&msk)
                break

    def rotate(self, i0, d0, i1):
        idx = self.idx
        n0, n1 = idx[i0], idx[i1]
        n2 = n3 = n1
        if n1.bal==-2:
            i2, n2 = n1.left, idx[n1.left]
        if n1.bal==2:
            i2, n2 = n1.right, idx[n1.right]
        if n2.bal==-1:
            i3, n3 = n2.left, idx[n2.left]
        if n2.bal==1:
            i3, n3 = n2.right, idx[n2.right]
        b1, b2, b3 = n1.bal, n2.bal, n3.bal
        
        if b1==-2 and b2==-1:
            n1.left = n2.right
            n2.right = i1
            n1.bal = n2.bal = 0
            nroot = i2
            
        if b1==2 and b2==1:
            n1.right = n2.left
            n2.left = i1
            n1.bal = n2.bal = 0
            nroot = i2

        if b1==-2 and b2==1:
            n1.left = n3.right
            n3.right = i1
            n2.right = n3.left
            n3.left = i2
            
            n3.bal = 0
            n1.bal = 1 if b3==-1 else 0
            n2.bal = -1 if b3==1 else 0
            nroot = i3

        if b1==2 and b2==-1:
            n1.right = n3.left
            n3.left = i1
            n2.left = n3.right
            n3.right = i2
            
            n3.bal = 0
            n1.bal = -1 if b3==1 else 0
            n2.bal = 1 if b3==-1 else 0
            nroot = i3
        
        if b1==-2 and b2==0:
            n1.left = n2.right
            n2.right = i1
            n2.bal = 1
            n1.bal = -1
            nroot = i2

        if b1==2 and b2==0:
            n1.right = n2.left
            n2.left = i1
            n2.bal = -1
            n1.bal = 1
            nroot = i2
            
        if i0==-1: self.root = nroot
        elif d0==-1: n0.left = nroot
        elif d0==1: n0.right = nroot
        return b2 == 0
        
    def pop(self, key):
        cur = parent = self.root
        msk = 0xffffffff>>1
        if cur==-1: return
        hist = self.hist
        n, deln = 0, 0

        while cur != -1:
            ilrk = self.idx[cur]
            ck = ilrk.key
            hist[n] = cur
            if key == ck:
                deln = n; break
            parent = cur
            if key < ck: # left
                cur = ilrk.left
            if key > ck: # right
                hist[n] |= 1<<31
                cur = ilrk.right # next
            n += 1
        if cur == -1: return # not found
        
        idx = self.idx
        node, pnode = idx[cur], idx[parent]
        
        if node.left == -1 and node.right == -1:
            if parent==cur:
                self.root = -1
            elif node.key < pnode.key:
                pnode.left = -1
            else: pnode.right = -1
            self.free(cur)
        elif node.left == -1:
            if parent==cur:
                self.root = node.right
            elif node.key < pnode.key:
                pnode.left = node.right
            else: pnode.right = node.right
            self.free(cur)
        elif node.right == -1:
            if parent==cur:
                self.root = node.left
            elif node.key < pnode.key:
                pnode.left = node.left
            else: pnode.right = node.left
            self.free(cur)
        else:
            hist[n] = cur | 1<<31
            n += 1
            sparent = scur = node.right
            while True:
                # print(idx[sparent].key, idx[scur].key)
                snode = idx[scur]
                if snode.left==-1: break
                hist[n] = scur
                n += 1
                sparent, scur = scur, snode.left
            
            snode.left, snode.bal = node.left, node.bal
            if sparent != scur:
                idx[sparent].left = snode.right
                snode.right = node.right
            hist[deln] = scur | 1<<31
            
            if parent==cur:
                self.root = scur
            elif node.key < pnode.key:
                pnode.left = scur
            else: pnode.right = scur
            self.free(cur)

        for i in range(n-1, -1, -1):
            n = hist[i]
            d = -1 if (n>>31)==0 else 1
            n &= msk
            
            idx[n].bal -=  d
            bal = idx[n].bal
            if bal==1 or bal==-1: break

            if bal==2 or bal==-2:
                rst = self.rotate(-1 if i==0 else hist[i-1]&msk,
                    -1 if (hist[i-1]>>31)==0 else 1, hist[i]&msk)
                if rst: break
        return self.body[cur]
    
    def alloc(self, k, val):
        self.size += 1
        cur = self.cur
        self.cur = self.idx[cur].id
        if self.size == self.cap:
            self.expand()
        self.body[cur] = val # [attr] self.body[cur]<-val
        ilrk = self.idx[cur]
        ilrk.left = -1
        ilrk.right = -1
        ilrk.key = k
        ilrk.bal = 0
        return cur
    
    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val): 
        self.push(key, val)
    
    def __len__(self):
        return self.size

    def free(self, idx):
        self.size -= 1
        self.idx[self.tail].id = idx
        self.tail = idx
        return self.body[idx]

def type_avl(ktype, dtype):
    ilr = np.dtype([('id', np.int32), ('left', np.int32), ('right', np.int32),
                ('key', ktype), ('bal', np.int8)])

    fields = [('idx', nb.from_dtype(ilr)[:]), ('root', nb.int32), ('cur', nb.int32),
              ('cap', nb.uint32), ('size', nb.uint32), ('tail', nb.uint32),
              ('hist', nb.uint32[:]), ('body', nb.from_dtype(dtype)[:])]

    local = {'ilr':ilr, 'dtype':dtype, 'ktype':ktype, 'np':np}

    subavl = sub_class(AVLTree, dtype)
    # print(subavl)
    exec(subavl, local)
    TypedAVLTree = local['TypedAVLTree']
    return nb.experimental.jitclass(fields)(TypedAVLTree)

class MemoryAVLTree:
    def __init__(self, cap=128, memory=None):
        self.avl = IntAVLTree(cap)
        self.memory = memory if memory is not None else typememory(cap)

    def push(self, key, val):
        self.avl.push(key, self.memory.push(val))

    def pop(self, key):
        idx = self.avl.pop(key)
        if idx is None: return
        return self.memory.pop(int(idx))

    def get(self, key):
        idx = self.avl.get(key)
        if idx is None: return
        return self.memory.body[int(idx)]

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val): 
        self.push(key, val)
    
    def __len__(self):
        return self.avl.size

    @property
    def size(self): return self.avl.size
    

def memory_avl(ktype, typememory):
    IntAVLTree = type_avl(ktype, np.int32)
    local = {'typememory': typememory, 'IntAVLTree': IntAVLTree}
    memoryavl = sub_class(MemoryAVLTree, None)
    # print(memorydeque)

    exec(memoryavl, local)
    TypedAVLTree = local['MemoryAVLTree']
    fields = [('avl', IntAVLTree.class_type.instance_type),
              ('memory', typememory.class_type.instance_type)]
    return nb.experimental.jitclass(fields)(TypedAVLTree)
    
def TypedAVLTree(ktype, dtype):
    if hasattr(dtype, 'class_type'):
        return memory_avl(ktype, dtype)
    else: return type_avl(ktype, dtype)
    
def print_tree(tree, mar=3, bal=False):
    nodes = [tree.root]
    rst = []
    while max(nodes)!=-1:
        cur = nodes.pop(0)
        if cur==-1:
            rst.append(' ')
            nodes.extend([-1, -1])
            continue
        ilrk = tree.idx[cur]
        value = ilrk['key']
        if bal: val = (val, ilrk['bal'])
        rst.append(value)
        nodes.append(ilrk['left'])
        nodes.append(ilrk['right'])

    def fmt(s, width):
        s = '%s'%str(s)
        l = len(s)
        pad = ' '*((width+1-l)//2)
        return (pad + s + pad)[:width]
    
    levels = int(np.ceil(np.log(len(rst)+1)/np.log(2)))       
    s = 0
    for r in range(levels):
        width = mar * 2 ** (levels - r - 1)
        line = [fmt(i, width) for i in rst[s:s+2**r]]
        print(''.join(line))
        s += 2 ** r
    print()

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    IntAVLTree = TypedAVLTree(np.uint32, np.uint32)
    tree = IntAVLTree(120)
    for i in [1,2,5,4,6,3,9,7,8]:
        tree.push(i, i)
    
    
    PointMemory = TypedMemory(t_point)
    PointAVLTree = TypedAVLTree(np.uint32, PointMemory)
    
    
    @nb.njit
    def test(point, x):
        for i in x: point.push(i, (0,0))

    x = np.arange(1024000, dtype=np.uint32)

    points = PointAVLTree(1024000+1)
    test(points, x)
    start = time()
    points = PointAVLTree(1024000+1)
    test(points, x)
    print(time()-start)
    
    
