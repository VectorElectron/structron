import numpy as np
import numba as nb

mode = 'map'

class RBTree:
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
            self.body = np.concatenate((self.body, self.body))
        
        idx[:self.cap] = self.idx
        self.idx = idx
        self.cur = self.cap
        self.cap *= 2
        self.tail = self.cap - 1

    def push(self, k, v=None):
        cur = parent = self.root
        
        if cur==-1:
            self.root = self.alloc(k, v)
            self.idx[self.root].bal = 1 # new black root
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
        if mode!='set':
            body = self.body
        n = 0

        # find the node
        while cur!=-1:
            ilrk = idx[cur]
            if mode!='func': ck = ilrk.key
            if mode=='func': ck = self.eval(body[cur])
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
        if mode != 'set':
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
            if mode!='func': node.key = snode.key
            if mode!='set':
                body[cur] = body[scur] # [if map]
            cur = scur
            node = snode
        
        # for i in hist[:n+1]: print(i, idx[i].key)

        # del black node with one red child
        if node.left != -1:
            if mode!='func': node.key = idx[node.left].key
            if mode!='set': body[cur] = body[node.left]
            self.free(node.left)
            node.left = -1
            if mode!='set': return value # [if map]
            return
        elif node.right != -1:
            if mode!='func': node.key = idx[node.right].key
            if mode!='set': body[cur] = body[node.right]
            self.free(node.right)
            node.right = -1
            if mode!='set': return value # [if map]
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
        if mode!='set': return value

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
            return self.body[idx] # [if map]

def istype(obj):
    if isinstance(obj, np.dtype): return True
    return isinstance(obj, type) and isinstance(np.dtype(obj), np.dtype)

def TypedRBTree(ktype, vtype=None):
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

    exec(inspect.getsource(RBTree), dict(globals()), locals())
    
    class TypedRBTree(locals()['RBTree']):
        _init_ = RBTree.__init__
        if mode=='func': eval = ktype
        def __init__(self, cap=16):
            self._init_(ilr, vtype, cap)
    
    return nb.experimental.jitclass(fields)(TypedRBTree)
    
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

def check_valid(tree, index=0, first=True):
    if first and tree[index]['bal']!=1: return False, 0
    if index == -1 or index >= len(tree):
        return True, 1  # NIL 节点是黑色的，且高度为 1

    node = tree[index]
    left_index = node['left']
    right_index = node['right']
    is_black = node['bal'] == 1

    if node['bal'] not in {0, 1}:
        return False, 0

    left_valid, left_black_height = check_valid(tree, left_index, 0)
    if not left_valid: return False, 0

    right_valid, right_black_height = check_valid(tree, right_index, 0)
    if not right_valid: return False, 0

    if left_black_height != right_black_height:
        return False, 0

    # 计算当前节点的黑色高度
    current_black_height = left_black_height + (1 if is_black else 0)

    if node['bal'] == 0:  # 当前节点是红色
        if left_index != -1 and tree[left_index]['bal'] == 0:
            return False, 0
        if right_index != -1 and tree[right_index]['bal'] == 0:
            return False, 0

    return True, current_black_height

if __name__ == '__main__':
    from time import time
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    
    PointRB = TypedRBTree(lambda self, p: p.x+p.y, t_point)
    points = PointRB()

    points.push(None, np.void((1,1), t_point))
    points.push(None, np.void((2,3), t_point))
    
    '''
    np.random.seed(1)
    x = np.arange(70000)
    np.random.shuffle(x)
    
    points = IntRedBlack()
    for i in x: points.push(i)
    for i in x[::2]: points.pop(i)

    
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
