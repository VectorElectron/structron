import numpy as np
import numba as nb

class Memory: # [rep] Memory->TypedMemory
    def __init__(self, cap=128, dtype=np.uint32): # [rep] , dtype=np.uint32->
        self.idx = np.arange(1, cap+1, dtype=np.uint32)
        self.cur = 0 # next blank
        self.cap = cap
        self.size = 0
        self.tail = cap-1
        self.body = np.zeros(cap, dtype)

    def expand(self):
        idx = np.arange(1, self.cap*2+1, dtype=np.uint32)
        self.body = np.concatenate((self.body, self.body))
        
        idx[:self.cap] = self.idx
        self.idx = idx
        self.cur = self.cap
        self.cap *= 2
        self.tail = self.cap - 1
    
    def push(self, obj):
        if self.size == self.cap:
            self.expand()
        self.size += 1
        cur = self.cur
        self.cur = self.idx[cur]
        self.body[cur] = obj # [attr] self.body[cur]<-obj
        return cur

    def __getitem__(self, index):
        return self.body[index]

    def __len__(self):
        return self.size

    def pop(self, idx):
        if self.size==self.cap:
            self.cur = idx
        self.size -= 1
        self.idx[self.tail] = idx
        self.tail = idx
        return self.body[idx]

import inspect
def sub_class(cls, dtype, **key):
    names = dtype.names if hasattr(dtype, 'names') else None
    code = inspect.getsource(cls)
    lines = []
    for line in code.split('\n'):
        if '[if' in line:
            k = line.split('[if')[1].split(']')[0][1:]
            if key.get(k, False)==False: continue
        if '[rep]' in line:
            what = line.split('[rep]')[1].strip()
            line = line.replace(*what.split('->'))
        elif '[swap]' in line:
            if not names is None:
                cur = line.split('[swap]')[1].strip()
                a, b = [i.strip() for i in cur.split('<->')]
                prefix = line[:line.index(a)]
                lines.append(prefix + '_a, _b = %s, %s'%(a, b))
                for i in names:
                    lines.append(prefix + '_a.%s, _b.%s = _b.%s, _a.%s'%(i,i,i,i))
                line = prefix + '# swap'
        elif '[attr]' in line:
            if not names is None:
                cur = line.split('[attr]')[1].strip()
                a, b = [i.strip() for i in cur.split('<-')]
                prefix = line[:line.index(a)]
                lines.append(prefix + '_ = ' + a)
                line = ['_.'+i for i in names]
                line = prefix + ','.join(line) + ' = '+ b
        lines.append(line)
    return '\n'.join(lines)

def TypedMemory(dtype):
    fields = [('idx', nb.uint32[:]), ('cur', nb.uint32),
              ('cap', nb.uint32), ('size', nb.uint32),
              ('tail', nb.uint32), ('body', nb.from_dtype(dtype)[:])]
    
    # def push(memory, x, y):
    local = {'dtype':dtype, 'np':np}

    submemory = sub_class(Memory, dtype)
    # print(submemory)
    exec(submemory, local)
    TypedMemory = local['TypedMemory']
    return nb.experimental.jitclass(fields)(TypedMemory)

if __name__ == '__main__':
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    PointMemory = TypedMemory(t_point)
    IntMemory = TypedMemory(np.uint32)
    points = PointMemory(2)
    lst = IntMemory(2)
    aaaa
    @nb.njit
    def test(points):
        for i in range(10240000):
            points.push((1,2))
        
    from time import time
    test(points)
    
    points = PointMemory(10240000)
    start = time()
    test(points)
    print(time()-start)
    
    
    '''
    p2 = points.push(np.void((2,1), t_point))
    print('size:', len(points))
    print('get', p1, points[p1])
    print('get', p2, points[p2])
    print('pop', p2, points.pop(p2))
    print('size:', len(points))
    '''
    
