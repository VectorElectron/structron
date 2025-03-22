import numba as nb
import numpy as np

emode = nmode = jitmode = 0

class Graph:
    def __init__(self, ktype, nidtype, eidtype, ntype=None, etype=None, cap=16):
        self.ncap = self.ecap = cap
        self.nsize = self.esize = 0
        self.ncur = self.ecur = 0
        self.nidx = np.zeros(cap, dtype=nidtype)
        self.eidx = np.zeros(cap, dtype=eidtype)
        self.nidx['id'][:] = np.arange(1, cap+1, dtype=np.int32)
        self.eidx['id'][:] = np.arange(1, cap+1, dtype=np.int32)

        if jitmode: self.nmap = nb.typed.Dict.empty(ktype, np.int32)
        else: self.nmap = {}
        
        if nmode: self.nbody = np.zeros(cap, dtype=ntype)
        if emode: self.ebody = np.zeros(cap, dtype=etype)
    
    def nexpand(self):
        nidx = np.concatenate((self.nidx, self.nidx))
        rg = np.arange(self.ncap+1, self.ncap*2+1, dtype=np.int32)
        if nmode: self.nbody = np.concatenate((self.nbody, self.nbody))

        nidx['id'][self.ncap:] = rg
        self.nidx = nidx
        self.ncur = self.ncap
        self.ncap *= 2
    
    def eexpand(self):
        eidx = np.concatenate((self.eidx, self.eidx))
        rg = np.arange(self.ecap+1, self.ecap*2+1, dtype=np.int32)
        if emode: self.ebody = np.concatenate((self.ebody, self.ebody))
        
        eidx['id'][self.ecap:] = rg
        self.eidx = eidx
        self.ecur = self.ecap
        self.ecap *= 2
    
    def add_node(self, key, node=None):
        if self.nsize == self.ncap:
            self.nexpand()
        self.nsize += 1
        cur = self.ncur
        self.nmap[key] = cur
        node = self.nidx[cur]
        self.ncur = node['id']
        node['key'] = key
        node['count'] = 0
        node['head'] = -1
        if nmode:
            self.nbody[cur] = node
        return cur

    def add_nodes(self, keys, nodes=None):
        for i in range(len(keys)):
            self.add_node(keys[i], nodes[i] if nmode else None)

    # did not remove the related edge auto, for it need iterate.
    def remove_node(self, idx, mode='key'):
        if mode=='key': idx = self.nmap[idx]
        if self.nsize==self.ncap:
            self.ncur = idx
        self.nsize -= 1

        node = self.nidx[idx]
        node['id'] = self.ncur
        self.ncur = idx
        self.nmap.pop(node['key'])

        eidx = self.eidx
        cur = node['head']

        if cur!=-1:
            while cur!=-1: cur = eidx[cur]['id']
            eidx[cur]['id'] = self.ecur
            self.ecur = node['head']
            self.esize -= node['count']
        node['count'] = -1
        
    
    def add_edge(self, start, end, edge=None):
        sidx = self.nmap[start]
        eidx = self.nmap[end]
        
        if self.esize == self.ecap:
            self.eexpand()
        self.esize += 1
        cur = self.ecur
        self.ecur = self.eidx[cur]['id']

        nid = self.nidx[sidx]
        eid = self.eidx[cur]
        
        eid['id'] = nid['head']
        nid['head'] = cur
        nid['count'] += 1
        eid['start'], eid['end'] = sidx, eidx

        if emode: self.ebody[cur] = edge
        return cur

    def add_edges(self, starts, ends, edges=None):
        for i in range(len(starts)):
            self.add_edge(starts[i], ends[i], edges[i] if emode else None)
    
    def remove_edge(self, start, end, mode='key'):
        if mode=='key':
            start, end = self.nmap[start], self.nmap[end]
            
        node = self.nidx[start]
        eidx = self.eidx
        cur = node['head']
        parent = -1
        found = 0
        while cur!=-1:
            ce = eidx[cur]
            if ce['end']==end:
                node['count'] -= 1
                if parent==-1:
                    node['head']=ce['id']
                else:
                    eidx[parent]['id'] = ce['id']
                found = 1
                break
            
            parent = cur
            cur = ce['id'] # next
        if not found: return 0
        
        idx = cur # removed idx
        if self.esize==self.ecap:
            self.ecur = idx
        self.esize -= 1
        self.eidx[idx]['id'] = self.ecur
        self.ecur = idx
        return 1 #self.body[idx]

    def remove_open_edge(self):
        nidx, eidx = self.nidx, self.eidx
        s = 0
        for idx in self.nmap.values():
            parent = -1
            node = nidx[idx]
            cur = node['head']
            while cur!=-1:
                ce = eidx[cur]
                if nidx[ce['end']]['count']<0:
                    s += 1
                    node['count'] -= 1
                    if parent==-1: node['head']=ce['id']
                    else: eidx[parent]['id'] = ce['id']
                        
                    if self.esize==self.ecap:
                        self.ecur = idx
                    self.esize -= 1
                    ocur = cur
                    cur = ce['id']
                    ce['id'] = self.ecur
                    self.ecur = ocur
                else:
                    parent = cur
                    cur = ce['id'] # next
        return s
    
    def neighbour(self, idx):
        eidx, nidx = self.eidx, self.nidx
        idx = self.nmap[idx]
        
        cur = nidx[idx]['head']
        while cur!=-1:
            e = eidx[cur]
            yield nidx[e['end']]['key']
            if mode=='body': yield e
            cur = e['id']

    def _neighbour(self, idx, mode='key'):
        eidx, nidx = self.eidx, self.nidx
        cur = nidx[idx]['head']
        while cur!=-1:
            yield cur            
            cur = eidx[cur]['id']

    def edges(self):
        nidx, eidx = self.nidx, self.eidx
        for idx in self.nmap.values():
            cur = nidx[idx]['head']
            while cur!=-1:
                e = eidx[cur]
                yield nidx[idx]['key'], nidx[e['end']]['key']
                cur = e['id']

    def _edges(self):
        nidx, eidx = self.nidx, self.eidx
        for idx in self.nmap.values():
            cur = nidx[idx]['head']
            while cur!=-1:
                e = eidx[cur]
                yield idx, e['end']
                cur = e['id']

    def nodes(self):
        nidx = self.nidx
        for k in self.nmap.keys(): yield k

    def _nodes(self):
        nidx = self.nidx
        for idx in self.nmap.values(): yield idx
        
    def lookup(self, key): return self.nmap[key]

    
def TypedGraph(ktype=None, ntype=None, etype=None, attr={}, jit=True):
    import inspect
    global nmode, emode, jitmode
    nmode = ntype is not None
    emode = etype is not None
    jitmode = jit

    nidtype = np.dtype([('id', np.int32), ('head', np.int32), ('count', np.int32), ('key', ktype)])
    
    eidtype = np.dtype([('id', np.int32), ('start', np.int32), ('end', np.int32)])
    
    fields = [('nidx', nb.from_dtype(nidtype)[:]), ('nsize', nb.int32),
              ('ncap', nb.uint32), ('ncur', nb.int32),
              ('eidx', nb.from_dtype(eidtype)[:]), ('esize', nb.int32),
              ('ecap', nb.uint32), ('ecur', nb.int32),
              ('nmap', nb.types.DictType(nb.from_dtype(ktype), nb.int32))]
    
    if nmode: fields.append(('nbody', nb.from_dtype(ntype)[:]))
    if emode: fields.append(('ebody', nb.from_dtype(etype)[:]))
    
    for k,v in attr.items(): fields.append((k, nb.from_dtype(v)))
    exec(inspect.getsource(Graph), dict(globals()), locals())
    
    class TypedGraph(locals()['Graph']):
        _init_ = Graph.__init__
        
        def __init__(self, cap=16):
            self._init_(ktype, nidtype, eidtype, ntype, etype, cap)

    if not jit: return TypedGraph
    return nb.experimental.jitclass(fields)(TypedGraph)

# only work for static (not removed or edited graph)
# @nb.njit
def print_graph(graph):
    nidx, eidx = graph.nidx, graph.eidx
    for i, cur in graph.nmap.items():
        print('node:', i)
        node = nidx[cur]
        cur = node['head']
        while cur!=-1:
            edge = eidx[cur]
            cur = edge['id']
            print(' ', node['key'], '-', nidx[edge['end']]['key'])

@nb.njit
def test(graph):
    for i in graph.nid_iter():
        print(i)
        
if __name__ == '__main__':
    t_edge = np.dtype([('weight', np.float32)]) 
    Graph = TypedGraph(ktype=np.int64, etype=None, jit=False)

    graph = Graph()
    for i in range(10):
        graph.add_node(i)

    for i in range(1,10):
        graph.add_edge(0, i)
    print_graph(graph)

    print(graph.eidx[:10])

    graph.remove_node(2)
    graph.remove_node(5)
    graph.remove_node(8)
    print(graph.remove_open_edge())
    print_graph(graph)
    
