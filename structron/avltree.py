import numpy as np
import numba as nb

mode = 'map'

class AVLTree:
    def __init__(self, ktype=np.int32, vtype=None, cap=16):
        self.idx = np.zeros(cap, dtype=ktype)
        if mode!='set':
            self.body = np.zeros(cap, dtype=vtype)
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
        idx = np.zeros(self.cap*2, dtype=self.idx.dtype)
        idx.id[:] = np.arange(1, self.cap*2+1, dtype=np.int32)
        if mode!='set':
            self.body = np.concatenate((self.body, self.body)) # [if map]
        
        idx[:self.cap] = self.idx
        self.idx = idx
        self.cur = self.cap
        self.cap *= 2
        self.tail = self.cap - 1
        
    def push(self, k, v=None):
        cur = parent = self.root
        if cur==-1:
            self.root = self.alloc(k, v)
            return
        
        idx = self.idx
        hist = self.hist
        dir = self.dir
        if mode=='func':
            body = self.body
            key = self.eval(v)
        if mode!='func': key = k
        n = 0
        
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            if mode!='func': ck = ilrk.key
            if mode=='func': ck = self.eval(body[cur])
            if key == ck:
                if mode!='set': self.body[cur] = v
                return
            
            hist[n] = cur
            if key < ck:
                cur = ilrk.left
                dir[n] = -1
            if key > ck:
                cur = ilrk.right
                dir[n] = 1
            n += 1
        
        hist[n] = self.alloc(k, v)
        idx = self.idx
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
        if mode!='set':
            body = self.body # [if map]
        n = 0

        # find the node
        while cur!=-1:
            ilrk = idx[cur]
            if mode!='func': ck = ilrk.key
            if mode=='func': ck = self.eval(body[cur])
            hist[n] = cur
            if key == ck: break
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
        if mode!='set':
            value = body[cur]
        
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
            if mode!='func': node.key = snode.key
            if mode!='set':
                body[cur] = body[scur] # [if map]
            cur = scur
            node = snode
            

        # del black node with one red child
        if node.left != -1:
            if mode!='func': node.key = idx[node.left].key
            if mode!='set': body[cur] = body[node.left]
            self.free(node.left)
            node.left = -1
            node.bal = 0
        elif node.right != -1:
            if mode!='func': node.key = idx[node.right].key
            if mode!='set': body[cur] = body[node.right]
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
        if mode!='set':
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
        idx = self.idx
        if mode!='set': body = self.body
        while cur != -1:
            node = idx[cur]
            if mode!='func': ck = node.key
            if mode=='func': ck = self.eval(body[cur])
            if key == ck:
                if mode!='set': return body[cur]
                return cur
            if key < ck: cur = node.left
            if key > ck: cur = node.right

    def has(self, key):
        cur = self.root
        if mode!='set': body = self.body
        while cur != -1:
            node = self.idx[cur]
            if mode!='func': ck = node.key
            if mode=='func': ck = self.eval(body[cur])
            if key == ck: return True
            if key < ck: cur = node.left
            if key > ck: cur = node.right
        return False

    def left(self, key):
        cur = parent = self.root        
        hist = self.hist
        dir = self.dir
        n = 0
        idx = self.idx
        if mode=='func':
            body = self.body
        
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            if mode!='func': ck = ilrk.key
            if mode=='func': ck = self.eval(body[cur])
            
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
                    if mode!='func': return lnode.key
                    return self.eval(body[nxt])
                nxt = lnode.right
        else:
            for i in range(n-1, -1, -1):
                if dir[i]==1:
                    # return self.body[hist[i]] # [if map]
                    if mode!='func': return idx[hist[i]].key
                    return self.eval(body[hist[i]])

    def right(self, key):
        cur = parent = self.root
        hist = self.hist
        dir = self.dir
        n = 0
        idx = self.idx
        if mode=='func':
            body = self.body
        
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            if mode!='func': ck = ilrk.key
            if mode=='func': ck = self.eval(body[cur])
            
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
                    if mode!='func': return lnode.key
                    return self.eval(body[nxt])
                nxt = lnode.left
        else:
            for i in range(n-1, -1, -1):
                if dir[i]==-1:
                    # return self.body[hist[i]] # [if map]
                    if mode!='func': return idx[hist[i]].key
                    return self.eval(body[hist[i]])
    
    def alloc(self, key, val=None):
        if self.size == self.cap:
            self.expand()
            
        self.size += 1
        cur = self.cur
        
        self.cur = self.idx[cur].id
        if mode!='set': self.body[cur] = val
            
        ilrk = self.idx[cur]
        ilrk.left = -1
        ilrk.right = -1
        ilrk.bal = 0
        if mode!='func': ilrk.key = key
        
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
        if mode!='set':
            return self.body[idx]
    
def istype(obj):
    if isinstance(obj, np.dtype): return True
    return isinstance(obj, type) and isinstance(np.dtype(obj), np.dtype)

def TypedAVLTree(ktype, vtype=None):
    import inspect
    global mode
    if not istype(ktype): mode = 'func'
    elif vtype is None: mode = 'set'
    else: mode = 'map'

    dtype = [('id', np.int32), ('left', np.int32),
        ('right', np.int32), ('key', ktype), ('bal', np.int8)]
    if mode=='func': dtype.pop(-2)
    ilr = np.dtype(dtype)
    
    
    fields = [('idx', nb.from_dtype(ilr)[:]), ('root', nb.int32), ('cur', nb.int32),
              ('cap', nb.uint32), ('size', nb.uint32), ('tail', nb.uint32),
              ('hist', nb.int32[:]), ('dir', nb.int32[:])]
    if vtype: fields.append(('body', nb.from_dtype(vtype)[:]))
    
    exec(inspect.getsource(AVLTree), dict(globals()), locals())
    
    class TypedAVLTree(locals()['AVLTree']):
        _init_ = AVLTree.__init__
        if mode=='func': eval = ktype
        def __init__(self, cap=16):
            self._init_(ilr, vtype, cap)
    
    return nb.experimental.jitclass(fields)(TypedAVLTree)
    
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

def check_valid(tree, index=0):
    if index == -1 or index >= len(tree):
        return True, 0  # NIL 节点高度为 0

    node = tree[index]
    left_index = node['left']
    right_index = node['right']
    key = node['key']
    bal = node['bal']

    left_valid, left_height = check_valid(tree, left_index)
    if not left_valid: return False, 0

    right_valid, right_height = check_valid(tree, right_index)
    if not right_valid: return False, 0

    if left_index != -1 and tree[left_index]['key'] >= key:
        return False, 0
    if right_index != -1 and tree[right_index]['key'] <= key:
        return False, 0

    current_height = max(left_height, right_height) + 1

    balance_factor = right_height - left_height
    if abs(balance_factor) > 1: return False, 0

    if bal != balance_factor: return False, 0
    return True, current_height

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    
    PointAVL = TypedAVLTree(lambda self, p: p.x+p.y, t_point)
    points = PointAVL()

    IntAVL = TypedAVLTree(np.int32, np.int32)
    ints = IntAVL()
    
    abcd
    @nb.njit
    def push_test(points, x):
        for i in x: points.push(i)

    @nb.njit
    def pop_test(points, x):
        for i in x: points.pop(i)
        
    
    np.random.seed(42)
    x = np.arange(1024000)
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
    
