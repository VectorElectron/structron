import numpy as np
import numba as nb
from .memory import sub_class, TypedMemory
import inspect

class RBTree: # [rep] RBTree->TypedRBTree
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
        
        if cur==-1:
            self.root = self.alloc(key, val)
            self.idx[self.root].bal = 1 # new black root
            return

        idx = self.idx
        hist = self.hist
        dir = self.dir
        n = 0
        
        while cur != -1:
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
        idx = self.idx
        
        pnode = idx[hist[n-1]]

        if dir[n-1]==1: pnode.right = hist[n]
        else: pnode.left = hist[n]
        
        
        while n>=0:
            
            if n==0: # root must be black
                idx[hist[n]].bal = 1
                break
                
            c_i = hist[n]
            f_i = hist[n-1]
            f_d = dir[n-1]
            
            if idx[f_i].bal == 1: break # father is black, need nothing

            gf_i = hist[n-2]
            gf_d = dir[n-2]
            gf_node = idx[gf_i]
            
            
            uc_n = gf_node.right if gf_d==-1 else gf_node.left
            uc_node = idx[uc_n] # get uncle
            
            if uc_n!=-1 and uc_node.bal == 0: # uncle is red
                uc_node.bal = 1
                gf_node.bal = 0
                idx[f_i].bal = 1
                n -= 2 # grand as current
                continue

            if key==40671: print('here', idx.size, self.size)
            
            if uc_n==-1 or uc_node.bal==1: # uncle is black
                # self.rotate(hist[n-3], dir[n-3], gf_i, gf_d, f_i, f_d, c_i)
                # break
            
                i0, b0, b1, b2 = hist[n-3], dir[n-3], gf_d, f_d
                n0, n1, n2, n3 = idx[i0], idx[gf_i], idx[f_i], idx[c_i]
                
                if b1==-1 and b2==-1: # left-left
                    n1.left = n2.right
                    n2.right = gf_i
                    n2.bal = 1
                    n1.bal = 0
                    nroot = f_i
                    
                if b1==1 and b2==1: # right-right
                    n1.right = n2.left
                    n2.left = gf_i
                    n2.bal = 1
                    n1.bal = 0
                    nroot = f_i

                if b1==-1 and b2==1: # left-right
                    n1.left = n3.right
                    n3.right = gf_i
                    n2.right = n3.left
                    n3.left = f_i
                    n3.bal = 1
                    n1.bal = 0
                    nroot = c_i

                if b1==1 and b2==-1: # right-left
                    n1.right = n3.left
                    n3.left = gf_i
                    n2.left = n3.right
                    n3.right = f_i
                    n3.bal = 1
                    n1.bal = 0
                    nroot = c_i
                    
                if i0==-1: self.root = nroot
                elif b0==-1: n0.left = nroot
                elif b0==1: n0.right = nroot
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
        
        # for i in hist[:n+1]: print(i, idx[i].key)

        # del black node with one red child
        if node.left != -1:
            node.key = idx[node.left].key
            body[cur] = body[node.left] # [if map]
            self.free(node.left)
            node.left = -1
            return value # [if map]
            return
        elif node.right != -1:
            node.key = idx[node.right].key
            body[cur] = body[node.right] # [if map]
            self.free(node.right)
            node.right = -1
            return value # [if map]
            return
        else:
            if n==0: self.root = -1
            elif dir[n-1]==-1: idx[hist[n-1]].left = -1
            elif dir[n-1]==1: idx[hist[n-1]].right = -1
        
        deln = hist[n]
        
        if n==0: self.root = -1
        elif dir[n-1]==-1: idx[hist[n-1]].left = -1
        elif dir[n-1]==1: idx[hist[n-1]].right = -1
        
        while n>0:
            nroot = 0
            stop = False
            c_i = hist[n]
            c_n = idx[c_i]

            f_i = hist[n-1]
            f_n = idx[f_i]
            
            g_n = idx[hist[n-2]]
            
            if c_n.bal==0 or n==0: # reach a red node or root
                c_n.bal = 1
                nroot = f_i
                stop = True
                
            elif dir[n-1]==-1: # del left node
                b_i = f_n.right
                b_n = idx[b_i]

                l_n = idx[b_n.left]
                r_n = idx[b_n.right]
            
                if b_n.bal == 0: # brother is red
                    #       F
                    # del /   \ B [r]
                    #      [b] / \ [b]
                    f_n.right = b_n.left
                    b_n.left = f_i
                    f_n.bal = 0; b_n.bal = 1
                    hist[n-1] = b_i
                    dir[n] = dir[n-1] = -1
                    hist[n] = f_i
                    hist[n+1] = c_i
        
                    if n==1: self.root = b_i
                    elif dir[n-2]==-1: g_n.left = b_i
                    else: g_n.right = b_i
                    n += 1
                    continue
                    
                elif b_n.bal==1: # brother is black
                    if b_n.right!=-1 and r_n.bal==0:
                        #       F
                        # del /   \ B
                        #            \ [r]
                        nroot = b_i
                        b_n.bal = f_n.bal
                        f_n.bal = r_n.bal = 1
                        f_n.right = b_n.left
                        b_n.left = f_i
                        stop = True
                        
                    elif b_n.left!=-1 and l_n.bal==0:
                        #       F
                        # del /   \ B
                        #      [r] /
                        nroot = b_n.left
                        l_n.bal = f_n.bal
                        f_n.bal = 1
                        f_n.right = l_n.left
                        b_n.left = l_n.right
                        l_n.left = f_i
                        l_n.right = b_i
                        stop = True
                    else:
                        #       F
                        # del /   \ B
                        #      [b] / \ [b]
                        b_n.bal = 0
                        n -= 1
                        continue

            elif dir[n-1]==1: # del right node
                b_i = f_n.left
                b_n = idx[b_i]

                l_n = idx[b_n.left]
                r_n = idx[b_n.right]
            
                if b_n.bal == 0: # brother is red
                    #              F
                    #      B [r] /   \ del
                    # [b] / \ [b]
                    f_n.left = b_n.right
                    b_n.right = f_i
                    f_n.bal = 0; b_n.bal = 1
                    hist[n-1] = b_i
                    dir[n] = dir[n-1] = 1
                    hist[n] = f_i
                    hist[n+1] = c_i

                    if n==1: self.root = b_i
                    elif dir[n-2]==-1: g_n.left = b_i
                    else: g_n.right = b_i
                    n += 1
                    continue
                    
                elif b_n.bal==1: # brother is black
                    if b_n.left!=-1 and l_n.bal==0:
                        #         F
                        #     B /   \ del
                        # [r]/
                        nroot = b_i
                        b_n.bal = f_n.bal
                        f_n.bal = l_n.bal = 1
                        f_n.left = b_n.right
                        b_n.right = f_i
                        stop = True
                    elif b_n.right!=-1 and r_n.bal==0:
                        #       F
                        #   B /   \ del
                        #    \ [r]
                        nroot = b_n.right
                        r_n.bal = f_n.bal
                        f_n.bal = 1
                        f_n.left = r_n.right
                        b_n.right = r_n.left
                        r_n.right = f_i
                        r_n.left = b_i
                        stop = True
                    else:
                        #          F
                        #      B /   \ del
                        # [b] / \ [b]
                        b_n.bal = 0
                        n -= 1
                        continue
            
            if n==1: self.root = nroot
            elif dir[n-2]==-1: g_n.left = nroot
            else: g_n.right = nroot
            if stop: break

        self.free(deln)
        return value # [if map]

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

def type_rb(ktype, vtype=None):
    ilr = np.dtype([('id', np.int32), ('left', np.int32), ('right', np.int32),
                ('key', ktype), ('bal', np.int8)])
    
    fields = [('idx', nb.from_dtype(ilr)[:]), ('root', nb.int32), ('cur', nb.int32),
              ('cap', nb.uint32), ('size', nb.uint32), ('tail', nb.uint32),
              ('hist', nb.int32[:]), ('dir', nb.int32[:])]

    if vtype: fields.append(('body', nb.from_dtype(vtype)[:]))
    
    local = {'ilr':ilr, 'vtype':vtype, 'ktype':ktype, 'np':np}

    subavl = sub_class(RBTree, vtype, map=vtype is not None)
    # print(subavl)
    exec(subavl, local)
    TypedRBTree = local['TypedRBTree']
    return nb.experimental.jitclass(fields)(TypedRBTree)

class MemoryRBTree:
    def __init__(self, cap=128, memory=None):
        self.avl = IntRBTree(cap)
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

def memory_rb(ktype, typememory):
    IntRBTree = type_rb(ktype, np.int32)
    local = {'typememory': typememory, 'IntRBTree': IntRBTree}
    memoryavl = sub_class(MemoryRBTree, None)
    # print(memorydeque)

    exec(memoryavl, local)
    TypedRBTree = local['MemoryRBTree']
    fields = [('avl', IntRBTree.class_type.instance_type),
              ('memory', typememory.class_type.instance_type)]
    return nb.experimental.jitclass(fields)(TypedRBTree)

def TypedRBTree(ktype, dtype=None):
    if hasattr(dtype, 'class_type'):
        return memory_rb(ktype, dtype)
    else: return type_rb(ktype, dtype)
    
def print_tree(tree, mar=4, bal=False):
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
        if ilrk['bal']: value = '('+str(value)+')'
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

def check_rb_tree(tree, index):
    # 如果当前节点是NIL节点（-1），则返回True和1（因为NIL节点是黑色的）
    if index == -1:
        return True, 1
    
    # 获取当前节点的信息
    node = tree[index]
    left_index = node['left']
    right_index = node['right']
    bal = node['bal']
    
    # 检查性质1：根节点必须是黑色的
    if index == 0 and bal != 1:
        return False, 0
    
    # 检查性质3：红色节点的子节点必须是黑色的
    if bal == 0:
        if left_index != -1 and tree[left_index]['bal'] != 1:
            return False, 0
        if right_index != -1 and tree[right_index]['bal'] != 1:
            return False, 0
    
    # 递归检查左子树和右子树
    left_valid, left_black_height = check_rb_tree(tree, left_index)
    right_valid, right_black_height = check_rb_tree(tree, right_index)
    
    # 如果左子树或右子树不合法，则整个树不合法
    if not left_valid or not right_valid:
        return False, 0
    
    # 检查性质4：从当前节点到叶子节点的所有路径的黑色节点数必须相同
    if left_black_height != right_black_height:
        return False, 0
    
    # 返回当前子树是否合法以及当前子树的黑色高度
    # 如果当前节点是黑色的，则黑色高度加1
    return True, left_black_height + (1 if bal == 1 else 0)

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    PointMemory = TypedMemory(t_point)
    points = PointMemory()
    
    IntRedBlack = TypedRBTree(np.int32)
    
    
    np.random.seed(1)
    x = np.arange(70000)
    np.random.shuffle(x)
    x = x[:65537]
    
    points = IntRedBlack()
    for i in range(len(x)):
        points.push(x[i])
    
    
    '''
    
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
    
    points = IntRedBlack(10240000+1)
    push_test(points, x[:3])
    pop_test(points, x[:3])

    # 0.32， 0.35
    points = IntRedBlack(10240000+1)
    a = time()
    push_test(points, x)
    b = time()
    pop_test(points, x)
    print(b-a, time()-b)
    '''
