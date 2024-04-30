import numpy as np
import numba as nb

class Memory:
    def __init__(self, cap=128, dtype=np.uint32):
        self.idx = np.arange(1, cap+1, dtype=np.uint32)
        self.cur = 0
        self.cap = cap
        self.size = 0
        self.tail = cap-1
        self.body = np.zeros(cap, dtype)

    def expand(self):
        idx = np.arange(1, self.cap*2+1, dtype=np.uint32)
        self.body = np.concatenate((self.body, self.body))
        
        idx[:self.cap] = self.idx
        self.idx = idx
        self.idx[self.tail] = self.cap
        self.cur = self.cap
        self.cap *= 2
        self.tail = self.cap - 1
    
    def push(self, obj):
        self.size += 1
        cur = self.cur
        self.cur = self.idx[cur]
        if self.size == self.cap:
            self.expand()
        self.body[cur] = obj
        return cur

    def __getitem__(self, index):
        return self.body[index]

    def __len__(self):
        return self.size

    def pop(self, idx):
        self.size -= 1
        self.idx[self.tail] = idx
        self.tail = idx
        return self.body[idx]

def make_names(dtype):
    if hasattr(dtype, 'names'):
        return ', '.join(dtype.names)
    return 'x'

def make_attrs(dtype, prefix):
    if hasattr(dtype, 'names'):
        namespair = [(i,i) for i in dtype.names]
        namespair = ['obj.%s = %s'%i for i in namespair]
        namespair.insert(0, 'obj = self.body[cur]')
        namespair = ('\n'+prefix).join(namespair)
        return namespair
    else: return 'self.body[cur] = x'
    
def TypedMemory(dtype):
    fields = [('idx', nb.uint32[:]), ('cur', nb.uint32),
              ('cap', nb.uint32), ('size', nb.uint32),
              ('tail', nb.uint32), ('body', nb.from_dtype(dtype)[:])]

    names, attrs = make_names(dtype), make_attrs(dtype, ' '*12)
    # def push(memory, x, y):
    local = {'Memory': Memory, 'dtype':dtype, 'np':np}
    
    submemory = '''
    class TypedMemory(Memory):
        def __init__(self, cap=128):
            self.idx = np.arange(1, cap+1, dtype=np.uint32)
            self.cur = 0
            self.cap = cap
            self.size = 0
            self.tail = cap-1
            self.body = np.zeros(cap, dtype)
        
        def push(self, %s):
            self.size += 1
            cur = self.cur
            self.cur = self.idx[cur]
            if self.size == self.cap:
                self.expand()
            %s
            return cur
    '''%(names, attrs)
    # print(submemory)
    
    submemory = '\n'.join([i[4:] for i in submemory.split('\n')])
    exec(submemory, local)
    TypedMemory = local['TypedMemory']
    return nb.experimental.jitclass(fields)(TypedMemory)

if __name__ == '__main__':
    t_point = np.dtype([('x', np.float32), ('y', np.float32)])
    PointMemory = TypedMemory(t_point)
    # IntMemory = type_memory(np.dtype(np.uint32))
    points = PointMemory(102400)

    @nb.njit
    def test(points):
        for i in range(102400000):
            x0 = points[0]
            points.body[i%102400] = x0
            
        
    from time import time
    test(points)
    points = PointMemory(102400)
    points.push(0, 5)
    
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
    
