import numpy as np

ilr = np.dtype([('id', np.int32), ('left', np.int32), ('right', np.int32), ('key', np.uint32)])

class BinarySearchTree:
    def __init__(self, cap=128, ktype=np.uint32, dtype=np.uint32):
        self.idx = np.rec.array(np.zeros(cap, dtype=ilr))
        self.idx['id'] = np.arange(1, cap+1, dtype=np.int32)
        self.root = -1
        self.cur = 0
        self.cap = cap
        self.size = 0
        self.tail = cap-1
        # self.key = np.zeros(cap, ktype)
        self.body = np.zeros(cap, dtype)

    def expand(self):
        idx = np.rec.array(np.zeros(self.cap*2, dtype=ilr))
        idx['id'] = np.arange(1, self.cap*2+1, dtype=np.int32)
        self.body = np.concatenate((self.body, self.body))
        # self.key = np.concatenate((self.key, self.key))
        idx[:self.cap] = self.idx
        self.idx = idx
        
        self.idx[self.tail].id = self.cap
        self.cur = self.cap
        self.cap *= 2
        self.tail = self.cap - 1

    def find(self, k):
        cur = self.root
        if cur==-1: return None
        while True:
            ilrk = self.idx[cur]
            ck = ilrk[cur].key
            if k == ck: # replace
                return cur
            if k < ck: # left
                if ilrk.left==-1: return None
                else: cur = ilrk.left # next
            if k > ck: # right
                if ilrk.right==-1: return None
                else: cur = ilrk.right # next
    
    def push(self, k, obj):
        cur = self.root
        if cur==-1:
            self.root = self.malloc(k, obj)
            return self.root
        while True:
            ilrk = self.idx[cur]
            ck = ilrk.key            
            if k == ck: # replace
                self.body[cur] = obj
            if k < ck: # left
                if ilrk.left==-1: # leaf
                    ilrk.left = self.cur
                    i = self.malloc(k, obj)
                    return i
                else: cur = ilrk.left # next
            if k > ck: # right
                if ilrk.right==-1: # leaf
                    ilrk.right = self.cur
                    i = self.malloc(k, obj)
                    return i
                else: cur = ilrk.right # next

    def pop(self, k):
        idx = self.idx
        cur = self.root # 删除根
        parent = lr = cur
        if cur==-1: return None
        while True:
            ilr = idx[cur]
            ck = self.key[cur]
            if k == ck: # replace
                break
            if k < ck: # left
                if ilr[1]==-1: return None
                else:
                    parent = cur
                    lr = 1
                    cur = ilr[1] # next
            if k > ck: # right
                if ilr[2]==-1: return None
                else:
                    parent = cur
                    lr = 2
                    cur = ilr[2] # next
        
        ilr = idx[cur]
        if ilr[1]==-1 and ilr[2]==-1:
            if parent==cur: self.root = -1
            else: idx[parent, lr] = -1
            self.free(cur)
        elif ilr[1]==-1:
            if parent==cur: self.root = ilr[2]
            idx[parent, lr] = ilr[2]
            self.free(cur)
        elif ilr[2]==-1:
            if parent==cur: self.root = ilr[1]
            idx[parent, lr] = ilr[1]
            self.free(cur)
        else:
            pleaf = cur
            oleaf = ilr[2]
            while True:
                left = idx[oleaf,1]
                if left==-1: break
                pleaf = oleaf
                oleaf = left
            self.key[cur] = self.key[oleaf]
            self.body[cur] = self.key[oleaf]

            idx[pleaf, (oleaf==ilr[2])+1] = idx[oleaf, 2]
            self.free(oleaf)
        
    def malloc(self, k, obj):
        self.size += 1
        cur = self.cur
        self.cur = self.idx[cur].id
        if self.size == self.cap:
            self.expand()
        self.body[cur] = obj
        ilrk = self.idx[cur]
        ilrk.left = -1
        ilrk.right = -1
        ilrk.key = k
        return cur

    def __getitem__(self, index):
        return self.body[index]

    def __len__(self):
        return self.size

    def free(self, idx):
        self.size -= 1
        self.idx[self.tail,0] = idx
        self.tail = idx
        return self.body[idx]

def print_tree(tree, mar=3):
    nodes = [tree.root]
    rst = []
    while max(nodes)>=0:
        cur = nodes.pop(0)
        if cur==-1:
            rst.append(' ')
            nodes.extend([-1, -1])
            continue
        ilrk = tree.idx[cur]
        rst.append(ilrk.key)
        nodes.append(ilrk.left)
        nodes.append(ilrk.right)

    def fmt(s, width):
        s = '%s'%s
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



if __name__ == '__main__':
    bst = BinarySearchTree(2)
    for i in (50, 30, 20):
        bst.push(i, i+1)

    print_tree(bst)
