import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core.extending import overload_method

@nb.njit(cache=True)
def expand(self):
    idx = np.zeros(self.cap*2, dtype=self.idx.dtype)
    idx.id[:] = np.arange(1, self.cap*2+1, dtype=np.int32)
    self.body = idx['body']
    idx[:self.cap] = self.idx
    self.idx = idx
    self.cur = self.cap
    self.cap *= 2
    self.tail = self.cap - 1

def build_push(mode='set'):
    @nb.njit(cache=True)
    def push(self, k, v=None):
        cur = parent = self.root
        if cur==-1:
            self.root = self.alloc(k, v)
            self.idx[self.root].bal = 1 # new black root
            return self.root

        idx = self.idx
        hist = self.hist
        dir = self.dir
        body = self.body
        
        if mode=='map': ek = k
        if mode=='set' or mode=='eval': ek = self.eval(k)
        
        n = 0
        
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            if mode=='comp': br = self.comp(k, body[cur])
            else: br = ek - self.eval(body[cur])

            if br==0: # found
                if mode=='map':
                    body[cur]['key'], body[cur]['value'] = k, v
                else: body[cur] = k
                return cur
            
            hist[n] = cur
            if br<0:
                cur = ilrk.left
                dir[n] = -1
            if br>0:
                cur = ilrk.right
                dir[n] = 1
            n += 1
        
        hist[n] = cur = self.alloc(k, v)
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
        return cur
    return push

push_set = build_push('set')
push_map = build_push('map')
push_eval = build_push('eval')
push_comp = build_push('comp')

def build_alloc(mode='map'):
    @nb.njit(cache=True)
    def alloc(self, k, v=None):
        if self.size == self.cap:
            self.expand()

        body = self.body
        self.size += 1
        cur = self.cur
        
        self.cur = self.idx[cur].id

        if mode=='map':
            body[cur]['key'], body[cur]['value'] = k, v
        else: body[cur] = k
            
        ilrk = self.idx[cur]
        ilrk.left = -1
        ilrk.right = -1
        ilrk.bal = 0
        return cur
    return alloc

alloc_map = build_alloc('map')
alloc_set = build_alloc('set')

def build_pop(mode='set'):
    @nb.njit(cache=True)
    def pop(self, k):
        parent = -1
        cur = self.root
        
        hist = self.hist
        dir = self.dir
        idx = self.idx
        body = self.body

        if mode=='map': ek = k
        if mode=='set' or mode=='eval': ek = self.eval(k)
        
        n = 0

        # find the node
        while cur!=-1:
            ilrk = idx[cur]

            if mode=='comp': br = self.comp(k, body[cur])
            else: br = ek - self.eval(body[cur])
            
            hist[n] = cur
            if br==0:
                break
            if br<0:
                cur = ilrk.left
                dir[n] = -1
            if br>0:
                cur = ilrk.right
                dir[n] = 1

            # print(idx[hist[n]].key, dir[n])
            n += 1
        
        # for i in hist[:n+1]: print(idx[i].key)
        if cur == -1: return # not found

        node = idx[cur]
        n0 = n
        self.buf[0] = body[cur]
        
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
            # if mode!='func': node.key = snode.key
            # if mode!='set': body[cur] = body[scur]

            snode.left, node.left = node.left, snode.left
            snode.right, node.right = node.right, snode.right
            snode.bal, node.bal = node.bal, snode.bal
            hist[n0], hist[n] = hist[n], hist[n0]
            
            if n0==0: self.root = scur
            elif dir[n0-1]==-1: idx[hist[n0-1]].left = scur
            elif dir[n0-1]==1: idx[hist[n0-1]].right = scur
            
            cur = scur
            # node = snode
        
        # del black node with one red child
        if node.left != -1: goal = node.left
        elif node.right != -1: goal = node.right
        else: goal = -1

        if n==0: self.root = goal
        elif dir[n-1]==-1: idx[hist[n-1]].left = goal
        elif dir[n-1]==1: idx[hist[n-1]].right = goal

        self.free(hist[n])
        
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

        if mode=='map': return self.buf[0]['value']
        else: return self.buf[0]
    return pop

pop_set = build_pop('set')
pop_map = build_pop('map')
pop_eval = build_pop('eval')
pop_comp = build_pop('comp')

@nb.njit(cache=True)
def free(self, idx):
    if self.size==self.cap:
        self.cur = idx
    self.size -= 1
    self.idx[self.tail].id = idx
    self.tail = idx
    return self.body[idx]

def build_index(mode):
    @nb.njit(cache=True)
    def index(self, k):
        cur = self.root
        idx = self.idx
        body = self.body

        if mode=='map': ek = k
        if mode=='set' or mode=='eval': ek = self.eval(k)
        
        while cur != -1:
            ilrk = idx[cur]
            if mode=='comp': br = self.comp(k, body[cur])
            else: br = ek - self.eval(body[cur])

            if br==0: return cur
            if br<0: cur = ilrk.left
            if br>0: cur = ilrk.right
        return -1
    return index

index_set = build_index('set')
index_map = build_index('map')
index_eval = build_index('eval')
index_comp = build_index('comp')

@nb.njit(cache=True)
def has(self, k):
    return self.index(k) >= 0
        
def build_left(mode='set'):
    @nb.njit(cache=True)
    def left(self, k):
        cur = self.root
        idx = self.idx
        hist = self.hist
        hist[-2] = 0 # level
        self.dir[-2] = -1
        body = self.body
        
        if mode=='map': ek = k
        if mode=='set' or mode=='eval': ek = self.eval(k)

        rst = -1
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            if mode=='comp': br = self.comp(k, body[cur])
            else: br = ek - self.eval(body[cur])
            
            if br<=0: # left
                cur = ilrk.left
            if br>0: # right
                rst = cur
                hist[hist[-2]] = cur
                hist[-2] += 1
                cur = ilrk.right
        return rst
    return left

left_set = build_left('set')
left_map = build_left('map')
left_eval = build_left('eval')
left_comp = build_left('comp')

def build_right(mode='set'):
    @nb.njit(cache=True)
    def right(self, k):
        cur = self.root
        idx = self.idx
        hist = self.hist
        hist[-2] = 0 # level
        self.dir[-2] = 1
        body = self.body
        
        if mode=='map': ek = k
        if mode=='set' or mode=='eval': ek = self.eval(k)

        rst = -1
        while cur != -1:
            parent = cur
            ilrk = idx[cur]
            if mode=='comp': br = self.comp(k, body[cur])
            else: br = ek - self.eval(body[cur])
            
            if br<0: # left
                rst = cur
                hist[hist[-2]] = cur
                hist[-2] += 1
                cur = ilrk.left
            if br>=0: # right
                cur = ilrk.right
        return rst
    return right

right_set = build_right('set')
right_map = build_right('map')
right_eval = build_right('eval')
right_comp = build_right('comp')

@nb.njit(cache=True)
def next(self):
    idx = self.idx
    hist = self.hist
    dir = self.dir[-2]

    while hist[-2] >0:
        hist[-2] -= 1
        cur = hist[hist[-2]]
        
        if dir==1: hist[-3] = idx[cur].right
        if dir==-1: hist[-3] = idx[cur].left

        while hist[-3]!=-1:
            hist[hist[-2]] = hist[-3]
            hist[-2] += 1
            if dir==1: hist[-3] = idx[hist[-3]].left
            if dir==-1: hist[-3] = idx[hist[-3]].right

        yield cur
    
@nb.njit(cache=True)
def min(self):
    rst = cur = self.root
    idx = self.idx
    hist = self.hist
    hist[-2] = 0 # level
    self.dir[-2] = 1

    while cur != -1:
        rst = cur
        hist[hist[-2]] = cur
        hist[-2] += 1
        ilrk = idx[cur]
        cur = ilrk.left
    return rst

@nb.njit(cache=True)
def max(self):
    rst = cur = self.root
    idx = self.idx
    hist = self.hist
    hist[-2] = 0 # level
    self.dir[-2] = -1

    while cur != -1:
        rst = cur
        hist[hist[-2]] = cur
        hist[-2] += 1
        ilrk = idx[cur]
        cur = ilrk.right
    return rst
        
@nb.njit(cache=True)
def __getitem__(self, k):
    idx = self.index(k)
    if idx>=0: return self.body[idx]['value']

@nb.njit(cache=True)
def __setitem__(self, k, v): 
    self.push(k, v)

@nb.njit(cache=True)
def __len__(self):
    return self.size

@nb.njit(cache=True)
def __lshift__(self, k):
    self.left(k)
    idx = self.idx
    hist = self.hist
    dir = self.dir[-2]

    while hist[-2] >0:
        hist[-2] -= 1
        cur = hist[hist[-2]]
        
        if dir==1: hist[-3] = idx[cur].right
        if dir==-1: hist[-3] = idx[cur].left

        while hist[-3]!=-1:
            hist[hist[-2]] = hist[-3]
            hist[-2] += 1
            if dir==1: hist[-3] = idx[hist[-3]].left
            if dir==-1: hist[-3] = idx[hist[-3]].right

        yield cur

@nb.njit(cache=True)
def __rshift__(self, k):
    self.right(k)
    idx = self.idx
    hist = self.hist
    dir = self.dir[-2]

    while hist[-2] >0:
        hist[-2] -= 1
        cur = hist[hist[-2]]
        
        if dir==1: hist[-3] = idx[cur].right
        if dir==-1: hist[-3] = idx[cur].left

        while hist[-3]!=-1:
            hist[hist[-2]] = hist[-3]
            hist[-2] += 1
            if dir==1: hist[-3] = idx[hist[-3]].left
            if dir==-1: hist[-3] = idx[hist[-3]].right

        yield cur

@nb.njit(cache=True)
def items(self):
    self.min()
    idx = self.idx
    hist = self.hist
    dir = self.dir[-2]

    while hist[-2] >0:
        hist[-2] -= 1
        cur = hist[hist[-2]]
        
        if dir==1: hist[-3] = idx[cur].right
        if dir==-1: hist[-3] = idx[cur].left

        while hist[-3]!=-1:
            hist[hist[-2]] = hist[-3]
            hist[-2] += 1
            if dir==1: hist[-3] = idx[hist[-3]].left
            if dir==-1: hist[-3] = idx[hist[-3]].right

        yield idx[cur]['body']

@nb.njit(cache=True)
def __iter__(self):
    self.min()
    idx = self.idx
    hist = self.hist
    dir = self.dir[-2]

    while hist[-2] >0:
        hist[-2] -= 1
        cur = hist[hist[-2]]
        
        if dir==1: hist[-3] = idx[cur].right
        if dir==-1: hist[-3] = idx[cur].left

        while hist[-3]!=-1:
            hist[hist[-2]] = hist[-3]
            hist[-2] += 1
            if dir==1: hist[-3] = idx[hist[-3]].left
            if dir==-1: hist[-3] = idx[hist[-3]].right

        yield cur

def istype(obj):
    if isinstance(obj, np.dtype): return True
    return isinstance(obj, type) and isinstance(np.dtype(obj), np.dtype)

def TypedRBTree(BaseStruct, BaseProxy, ktype=None, vtype=None, attrs={}):
    import inspect
    if ktype is None:
        mode = 'set'
        ktype = lambda self, x:x
    if not istype(ktype):
        n = len(inspect.signature(ktype).parameters)
        mode = 'eval' if n==2 else 'comp'
    else:
        vtype = np.dtype([('key', ktype), ('value', vtype)])
        mode, ktype = 'map', lambda self, x: x['key']

    struct = structref.register(BaseStruct)


    idxtype = np.dtype([('id', np.int32), ('left', np.int32),
        ('right', np.int32), ('bal', np.int8), ('body', vtype)])
    
    t_heap = struct([('idx', nb.from_dtype(idxtype)[:]),
            ('body', nb.from_dtype(vtype)[:]),
            ('buf', nb.from_dtype(vtype)[:]), ('root', nb.int32),
            ('cur', nb.int32), ('cap', nb.uint32),
            ('size', nb.uint32), ('tail', nb.uint32),
            ('hist', nb.int32[:]), ('dir', nb.int32[:])] +
           [(k, nb.from_dtype(v)) for k,v in attrs.items()])

    push = {'set': push_set, 'comp': push_comp, 'map': push_map, 'eval': push_eval}[mode]
    pop = {'comp': pop_comp, 'set': pop_set, 'map': pop_map, 'eval': pop_eval}[mode]
    index = {'comp': index_comp, 'set': index_set, 'map': index_map, 'eval': index_eval}[mode]
    left = {'comp': left_comp, 'set': left_set, 'map': left_map, 'eval': left_eval}[mode]
    right = {'comp': right_comp, 'set': right_set, 'map': right_map, 'eval': right_eval}[mode]

    alloc = (alloc_set, alloc_map)[mode=='map']
    
    temp = ''' # add attrs
        def get_%s(self): return self.%s
        def set_%s(self, v): self.%s = v
        BaseProxy.%s = property(nb.njit(get_%s), nb.njit(set_%s))
        del get_%s, set_%s'''
    temp = '\n'.join([i.strip() for i in temp.split('\n')])
    
    for i in ('idx', 'body', 'buf', 'root', 'cur', 'cap', 'size', 'tail'): exec(temp%((i,)*9))
    for i in attrs: exec(temp%((i,)*9))
    
    #BaseProxy.expand = expand
    BaseProxy.push = push
    BaseProxy.alloc = alloc
    BaseProxy.pop = pop
    BaseProxy.index = index
    BaseProxy.has = has
    BaseProxy.left = left
    BaseProxy.right = right
    BaseProxy.min = min
    BaseProxy.max = max
    BaseProxy.next = next
    BaseProxy.__getitem__ = __getitem__
    BaseProxy.__setitem__ = __setitem__
    BaseProxy.__len__ = __len__
    BaseProxy.__iter__ = __iter__
    BaseProxy.__lshift__ = __lshift__
    BaseProxy.__rshift__ = __rshift__
    BaseProxy.items = items

    structref.define_boxing(struct, BaseProxy)
    
    overload_method(struct, "alloc")(lambda self, k, v=None: alloc.py_func)    
    overload_method(struct, "push")(lambda self, k, v=None: push.py_func)
    overload_method(struct, "free")(lambda self, idx: free.py_func)
    overload_method(struct, "pop")(lambda self, k: pop.py_func)
    overload_method(struct, "min")(lambda self: min.py_func)
    overload_method(struct, "max")(lambda self: max.py_func)
    overload_method(struct, "next")(lambda self: next.py_func)
    overload_method(struct, "expand")(lambda self: expand.py_func)
    overload_method(struct, "index")(lambda self, k: index.py_func)
    overload_method(struct, "has")(lambda self, k: has.py_func)
    overload_method(struct, "left")(lambda self, k: left.py_func)
    overload_method(struct, "right")(lambda self, k: right.py_func)
    overload_method(struct, "rotate")(lambda self, i0, b0, i1, b1, i2, b2, i3: rotate.py_func)

    if mode!='comp': overload_method(struct, 'eval')(lambda self, x: ktype)
    else: overload_method(struct, 'comp')(lambda self, x1, x2: ktype)
    
    
    @nb.njit(cache=True)
    def init(cap=16):
        self = structref.new(t_heap)
        self.idx = np.zeros(cap, dtype=idxtype)
        self.body = self.idx['body'] # np.zeros(cap, dtype=vtype)
        self.buf = np.zeros(1, dtype=vtype)
        self.idx.id[:] = np.arange(1, cap+1, dtype=np.int32)
        self.root = -1
        self.cur = 0
        self.cap = cap
        self.size = 0
        self.tail = cap-1

        self.hist = np.zeros(256, dtype=np.int32)
        self.hist[-1] = -1
        self.dir = np.zeros(256, dtype=np.int32)
        return self

    def __new__(cls, cap=16):
        return init(cap)
    
    BaseProxy.__new__ = __new__
    return init
    
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
    class IntRBStruct(nb.types.StructRef): pass
    class IntRBProxy(structref.StructRefProxy): pass

    IntAVL = TypedRBTree(IntRBStruct, IntRBProxy, np.int32, np.int32)
    ints = IntAVL(16)
    aaaa
    
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    
    def f(self, x, y): return x-y
    
    IntRB = TypedRBTree(f, np.int32)
    ints = IntRB()
    
    x = np.arange(100)
    np.random.seed(0)
    np.random.shuffle(x)
    
    for i in x: ints.push(i)

    for i in x[::2]: ints.pop(i)
    print(ints.size)
    # print_tree(ints)
    abcd
    
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
