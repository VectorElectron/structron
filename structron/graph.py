import numpy as np
import numba as nb
from numba.experimental import structref
from numba.core.extending import overload_method

@nb.njit(cache=True)
def nexpand(self):
    nidx = np.concatenate((self.nidx, self.nidx))
    rg = np.arange(self.ncap+1, self.ncap*2+1, dtype=np.int32)
    nidx['id'][self.ncap:] = rg
    self.nidx = nidx
    self.ncur = self.ncap
    self.ncap *= 2

@nb.njit(cache=True)
def eexpand(self):
    eidx = np.concatenate((self.eidx, self.eidx))
    rg = np.arange(self.ecap+1, self.ecap*2+1, dtype=np.int32)

    eidx['id'][self.ecap:] = rg
    self.eidx = eidx
    self.ecur = self.ecap
    self.ecap *= 2

def build_add_node(nmode):
    @nb.njit(cache=True)
    def add_node(self, key, node=None):
        if self.nsize == self.ncap:
            nexpand(self)
        self.nsize += 1
        cur = self.ncur
        self.nmap[key] = cur
        node = self.nidx[cur]
        self.ncur = node['id']
        node['key'] = key
        node['count'] = 0
        node['head'] = -1
        if nmode: self.nbody[cur] = node
        return cur
    return add_node

add_node = build_add_node(False)
add_node_body = build_add_node(True)

def build_add_nodes(nmode):
    @nb.njit(cache=True)
    def add_nodes(self, keys, nodes=None):
        for i in range(len(keys)):
            if nmode: self.add_node(keys[i], nodes[i])
            else: self.add_node(keys[i], None)
    return add_nodes

add_nodes = build_add_nodes(False)
add_nodes_body = build_add_nodes(True)

@nb.njit(cache=True)
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

def build_add_edge(emode):
    @nb.njit(cache=True)
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

        if emode: eid['ebody'] = edge
        return cur
    return add_edge

add_edge = build_add_edge(False)
add_edge_body = build_add_edge(True)

def build_add_edges(emode):
    @nb.njit(cache=True)
    def add_edges(self, starts, ends, edges=None):
        for i in range(len(starts)):
            if emode: self.add_edge(starts[i], ends[i], edges[i])
            else: self.add_edge(starts[i], ends[i], None)
    return add_edges
            
add_edges = build_add_edges(False)
add_edges_body = build_add_edges(True)

@nb.njit(cache=True)
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

@nb.njit(cache=True)
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

@nb.njit(cache=True)
def neighbour(self, idx, mode='id'):
    eidx, nidx = self.eidx, self.nidx
    if mode=='key': idx = self.nmap[idx]
    cur = nidx[idx]['head']
    while cur!=-1:
        yield eidx[cur]            
        cur = eidx[cur]['id']

@nb.njit(cache=True)
def edges(self):
    nidx, eidx = self.nidx, self.eidx
    for idx in self.nmap.values():
        cur = nidx[idx]['head']
        while cur!=-1:
            e = eidx[cur]
            yield e
            cur = e['id']

@nb.njit(cache=True)
def nodes(self):
    nidx = self.nidx
    for i in self.nmap.values(): yield nidx[i]
        
def lookup(self, key): return self.nmap[key]


def TypedGraph(BaseStruct, BaseProxy, ktype=None, ntype=None, etype=None, weight=None, attrs={}):
    nmode = ntype is not None
    emode = etype is not None
    if weight is None: weight = lambda self, x: x

    nidtype = np.dtype([('id', np.int32), ('head', np.int32), ('count', np.int32), ('key', ktype)] +
                        ([] if ntype is None else [('nbody', ntype)]))
    
    eidtype = np.dtype([('id', np.int32), ('start', np.int32), ('end', np.int32)] +
                        ([] if etype is None else [('ebody', etype)]))

    struct = structref.register(BaseStruct)

    t_heap = struct([('nidx', nb.from_dtype(nidtype)[:]), ('nsize', nb.int32),
                     ('ncap', nb.uint32), ('ncur', nb.int32),
                     ('eidx', nb.from_dtype(eidtype)[:]), ('esize', nb.int32),
                     ('ecap', nb.uint32), ('ecur', nb.int32),
                     ('nmap', nb.types.DictType(nb.from_dtype(ktype), nb.int32))] +
                    [(k, nb.from_dtype(v)) for k,v in attrs.items()])
    
    
    temp = ''' # add attrs
        def get_%s(self): return self.%s
        def set_%s(self, v): self.%s = v
        BaseProxy.%s = property(nb.njit(get_%s), nb.njit(set_%s))
        del get_%s, set_%s'''
    temp = '\n'.join([i.strip() for i in temp.split('\n')])

    for i in ('nidx', 'nsize', 'ncap', 'ncur',
              'eidx', 'esize', 'ecap', 'ecur', 'nmap'): exec(temp%((i,)*9))
    for i in attrs: exec(temp%((i,)*9))

    _add_node = (add_node, add_node_body)[nmode]
    _add_nodes = (add_nodes, add_nodes_body)[nmode]
    _add_edge = (add_edge, add_edge_body)[emode]
    _add_edges = (add_edges, add_edges_body)[emode]

    
    BaseProxy.add_node = _add_node
    BaseProxy.add_nodes = _add_nodes
    BaseProxy.add_edge = _add_edge
    BaseProxy.add_edges = _add_edges
    
    BaseProxy.nexpand = nexpand
    BaseProxy.eexpand = eexpand
    BaseProxy.remove_node = remove_node
    BaseProxy.remove_edge = remove_edge
    BaseProxy.remove_open_edge = remove_open_edge
    BaseProxy.edges = edges
    BaseProxy.nodes = nodes
    BaseProxy.neighbour = neighbour
    BaseProxy.weight = nb.njit(weight)
    
    structref.define_boxing(struct, BaseProxy)
    
    overload_method(struct, "nexpand")(lambda self: nexpand.py_func)
    overload_method(struct, "eexpand")(lambda self: eexpand.py_func)
    overload_method(struct, "add_node")(lambda self, key, node=None: _add_node.py_func)
    overload_method(struct, "add_nodes")(lambda self, keys, nodes=None: _add_nodes.py_func)
    overload_method(struct, "remove_node")(lambda self: remove_node.py_func)
    overload_method(struct, "add_edge")(lambda self, start, end, edge=None: _add_edge.py_func)
    overload_method(struct, "add_edges")(lambda self, starts, ends, edges=None: _add_edges.py_func)
    overload_method(struct, "remove_edge")(lambda self: remove_edge.py_func)
    overload_method(struct, "remove_open_edge")(lambda self: remove_open_edge.py_func)
    overload_method(struct, "edges")(lambda self: edges.py_func)
    overload_method(struct, "nodes")(lambda self: nodes.py_func)
    overload_method(struct, "neighbour")(lambda self, idx, mode='id': neighbour.py_func)
    overload_method(struct, "weight")(lambda self, x: weight)

    @nb.njit(cache=True)
    def init(cap=16):
        self = structref.new(t_heap)
        self.ncap = self.ecap = cap
        self.nsize = self.esize = 0
        self.ncur = self.ecur = 0
        self.nidx = np.zeros(cap, dtype=nidtype)
        self.eidx = np.zeros(cap, dtype=eidtype)
        self.nidx['id'][:] = np.arange(1, cap+1, dtype=np.int32)
        self.eidx['id'][:] = np.arange(1, cap+1, dtype=np.int32)
        self.nmap = nb.typed.Dict.empty(ktype, np.int32)
        return self

    def __new__(cls, cap=16):
        return init(cap)
    
    BaseProxy.__new__ = __new__
    return init


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
    class GraphStruct(nb.types.StructRef): pass
    class GraphProxy(structref.StructRefProxy): pass

    t_edge = np.dtype([('weight', np.float32)]) 
    Graph = TypedGraph(GraphStruct, GraphProxy, ktype=np.int64, etype=np.int32)

    graph = Graph(16)

    for i in range(10):
        graph.add_node(i)


    for i in range(1,10):
        graph.add_edge(0, i, 10)
    print_graph(graph)
    
    aaaa
    print(graph.eidx[:10])

    graph.remove_node(2)
    graph.remove_node(5)
    graph.remove_node(8)
    print(graph.remove_open_edge())
    print_graph(graph)
    
    
