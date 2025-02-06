import numpy as np
import numba as nb
from .memory import sub_class, TypedMemory
import inspect

class AVLTree: # [rep] AVLTree->TypedAVLTree
    def __init__(self, cap=128, ktype=np.int32, vtype=None): # [rep] , ktype=np.int32, vtype=None->
        self.idx = np.zeros(cap, dtype=ilr)
        self.body = np.zeros(cap, dtype=vtype) # [if map]
        self.idx.id[:] = np.arange(1, cap+1, dtype=np.int32)
        # for i in range(cap): self.idx[i].id = i+1
        self.root = -1
        self.cur = 0
        self.cap = cap
        self.size = 0
        self.tail = cap-1

        self.hist = np.zeros(256, dtype=np.int32)
        self.hist[-1] = -1
        self.dir = np.zeros(256, dtype=np.int32)

    def expand(self):
        idx = np.zeros(self.cap*2, dtype=ilr)
        idx.id[:] = np.arange(1, self.cap*2+1, dtype=np.int32)
        self.body = np.concatenate((self.body, self.body)) # [if map]
        
        idx[:self.cap] = self.idx
        self.idx = idx
        self.cur = self.cap
        self.cap *= 2
        self.tail = self.cap - 1

    def push(self, key, val=None):
        cur = parent = self.root
        idx = self.idx
        if cur==-1:
            self.root = self.alloc(key, val)
            return
        
        hist = self.hist
        dir = self.dir
        n = 0
        
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            ck = ilrk.key
            if key == ck:
                self.body[cur] = val # [if map][attr] self.body[cur]<-val
                return
            
            hist[n] = cur
            if key < ck:
                cur = ilrk.left
                dir[n] = -1
            if key > ck:
                cur = ilrk.right
                dir[n] = 1
            n += 1
            
        hist[n] = self.alloc(key, val)
        pnode = idx[hist[n-1]]

        if dir[n-1]==1: pnode.right = hist[n]
        else: pnode.left = hist[n]

        for i in range(n-1, -1, -1):
            n = hist[i]
            d = dir[i]

            idx[n].bal += d
            bal = idx[n].bal

            if bal==0: break
            if bal==2 or bal==-2:
                self.rotate(hist[i-1], dir[i-1],
                    hist[i], dir[i], hist[i+1], dir[i+1], hist[i+2])
                break

    def pop(self, key):
        parent = -1
        cur = self.root
        
        hist = self.hist
        dir = self.dir
        idx = self.idx
        body = self.body # [if map]
        n = 0

        # find the node
        while cur!=-1:
            ilrk = idx[cur]
            ck = ilrk.key
            hist[n] = cur
            if key == ck:
                break
            if key < ck:
                cur = ilrk.left
                dir[n] = -1
            if key > ck:
                cur = ilrk.right
                dir[n] = 1

            # print(idx[hist[n]].key, dir[n])
            n += 1
        
        # for i in hist[:n+1]: print(idx[i].key)
        if cur == -1: return # not found

        node = idx[cur]
        value = body[cur] # [if map]
        
        # pnode = idx[hist[n-1]]
        
        # find the post next one
        if node.left!=-1 and node.right!=-1:
            dir[n] = 1
            n += 1
            scur = node.right
            while True:
                snode = idx[scur]
                hist[n] = scur
                if snode.left==-1: break
                dir[n] = -1
                # print(idx[hist[n]].key, dir[n])
                n += 1
                scur = snode.left
            node.key = snode.key
            body[cur] = body[scur] # [if map]
            cur = scur
            node = snode
            

        # del black node with one red child
        if node.left != -1:
            node.key = idx[node.left].key
            body[cur] = body[node.left] # [if map]
            self.free(node.left)
            node.left = -1
            node.bal = 0
        elif node.right != -1:
            node.key = idx[node.right].key
            body[cur] = body[node.right] # [if map]
            self.free(node.right)
            node.right = -1
            node.bal = 0
        else:
            if n==0: self.root = -1
            elif dir[n-1]==-1: idx[hist[n-1]].left = -1
            elif dir[n-1]==1: idx[hist[n-1]].right = -1
            self.free(hist[n])

        for i in range(n-1, -1, -1):
            n = hist[i]
            d = dir[i]
            c_n = idx[n]
            c_n.bal -= d
            bal = c_n.bal
            
            if bal==1 or bal==-1: break
            if bal==2 or bal==-2:
                # if self.rotate(hist[i-1], dir[i-1], hist[i]): break
                i2 = c_n.right if dir[i]==-1 else c_n.left
                n2 = idx[i2]
                stop = n2.bal==0
                i3 = n2.right if n2.bal==1 else n2.left
                self.rotate(hist[i-1], dir[i-1],
                    hist[i], -dir[i], i2, n2.bal, i3)
                if stop: break
        return self.body[cur] # [if map]
        
    def rotate(self, i0, b0, i1, b1, i2, b2, i3):
        idx = self.idx
        n0, n1, n2, n3 = idx[i0], idx[i1], idx[i2], idx[i3]

        if b1==-1 and b2==-1:
            n1.left = n2.right
            n2.right = i1
            n1.bal = n2.bal = 0
            nroot = i2
            
        if b1==1 and b2==1:
            n1.right = n2.left
            n2.left = i1
            n1.bal = n2.bal = 0
            nroot = i2

        b3 = n3.bal
        if b1==-1 and b2==1:
            n1.left = n3.right
            n3.right = i1
            n2.right = n3.left
            n3.left = i2
            
            n3.bal = 0
            n1.bal = 1 if b3==-1 else 0
            n2.bal = -1 if b3==1 else 0
            nroot = i3
            
        if b1==1 and b2==-1:
            n1.right = n3.left
            n3.left = i1
            n2.left = n3.right
            n3.right = i2
            
            n3.bal = 0
            n1.bal = -1 if b3==1 else 0
            n2.bal = 1 if b3==-1 else 0
            nroot = i3
        
        if b1==-1 and b2==0:
            n1.left = n2.right
            n2.right = i1
            n2.bal = 1
            n1.bal = -1
            nroot = i2

        if b1==1 and b2==0:
            n1.right = n2.left
            n2.left = i1
            n2.bal = -1
            n1.bal = 1
            nroot = i2
        
        if i0==-1: self.root = nroot
        elif b0==-1: n0.left = nroot
        elif b0==1: n0.right = nroot
        return b2 == 0
    
    def get(self, key):
        cur = self.root
        while cur != -1:
            node = self.idx[cur]
            ck = node.key
            if key == ck:
                return self.body[cur] # [if map]
                return cur
            if key < ck: cur = node.left
            if key > ck: cur = node.right

    def has(self, key):
        cur = self.root
        while cur != -1:
            node = self.idx[cur]
            ck = node.key
            if key == ck:
                return True
            if key < ck: cur = node.left
            if key > ck: cur = node.right
        return False

    def left(self, key):
        cur = parent = self.root        
        hist = self.hist
        dir = self.dir
        n = 0
        idx = self.idx
        
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            ck = ilrk.key
            if key == ck: break
            if key < ck: # left
                hist[n] = cur
                dir[n] = -1
                cur = ilrk.left
            if key > ck: # right
                hist[n] = cur
                dir[n] = 1
                cur = ilrk.right
            n += 1
        if cur == -1: return
        node = idx[cur]
        if node.left != -1:
            nxt = node.left
            while True:
                lnode = idx[nxt]
                if lnode.right==-1:
                    # return self.body[nxt] # [if map]
                    return lnode.key
                nxt = lnode.right
        else:
            for i in range(n-1, -1, -1):
                if dir[i]==1:
                    # return self.body[hist[i]] # [if map]
                    return idx[hist[i]].key

    def right(self, key):
        cur = parent = self.root
        hist = self.hist
        dir = self.dir
        n = 0
        idx = self.idx
        
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            ck = ilrk.key
            if key == ck: break
            if key < ck: # left
                hist[n] = cur
                dir[n] = -1
                cur = ilrk.left
            if key > ck: # right
                hist[n] = cur
                dir[n] = 1
                cur = ilrk.right
            n += 1
        if cur == -1: return
        node = idx[cur]
        if node.right != -1:
            nxt = node.right
            while True:
                lnode = idx[nxt]
                if lnode.left==-1:
                    # return self.body[nxt] # [if map]
                    return lnode.key
                nxt = lnode.left
        else:
            for i in range(n-1, -1, -1):
                if dir[i]==-1:
                    # return self.body[hist[i]] # [if map]
                    return idx[hist[i]].key
        
    def alloc(self, key, val=None):
        if self.size == self.cap:
            self.expand()
        
        self.size += 1
        cur = self.cur
        
        self.cur = self.idx[cur].id
        self.body[cur] = val # [if map][attr] self.body[cur]<-val
        ilrk = self.idx[cur]
        ilrk.left = -1
        ilrk.right = -1
        ilrk.key = key
        ilrk.bal = 0
        return cur
    
    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val): 
        self.push(key, val)
    
    def __len__(self):
        return self.size

    def __lshift__(self, key):
        return self.left(key)

    def __rshift__(self, key):
        return self.right(key)

    def free(self, idx):
        if self.size==self.cap:
            self.cur = idx
        self.size -= 1
        self.idx[self.tail].id = idx
        self.tail = idx
        return self.body[idx] # [if map]

def type_avl(ktype, vtype=None):
    ilr = np.dtype([('id', np.int32), ('left', np.int32), ('right', np.int32),
                ('key', ktype), ('bal', np.int8)])
    
    fields = [('idx', nb.from_dtype(ilr)[:]), ('root', nb.int32), ('cur', nb.int32),
              ('cap', nb.uint32), ('size', nb.uint32), ('tail', nb.uint32),
              ('hist', nb.int32[:]), ('dir', nb.int32[:])]

    if vtype: fields.append(('body', nb.from_dtype(vtype)[:]))
    
    local = {'ilr':ilr, 'vtype':vtype, 'ktype':ktype, 'np':np}

    subavl = sub_class(AVLTree, vtype, map=vtype is not None)
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

    def has(self, key): return self.avl.has(key)

    def left(self, key): return self.avl.left(key)

    def right(self, key): return self.avl.right(key)
    
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

def TypedAVLTree(ktype, dtype=None):
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
        if bal: value = (value, ilrk['bal'])
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
    PointMemory = TypedMemory(t_point)
    points = PointMemory()
    
    IntAVL = TypedAVLTree(np.int32)

    
    points = IntAVL()

    @nb.njit
    def push_test(points, x):
        for i in x: points.push(i)

    @nb.njit
    def pop_test(points, x):
        for i in x: points.pop(i)
        
    
    np.random.seed(42)
    x = np.arange(10240000)
    np.random.shuffle(x)

    # np.random.shuffle(x)
    
    points = IntAVL(10240000+1)
    push_test(points, x[:3])
    pop_test(points, x[:3])

    points = IntAVL(10240000+1)
    a = time()
    push_test(points, x)
    b = time()
    pop_test(points, x)
    print(b-a, time()-b)
    
