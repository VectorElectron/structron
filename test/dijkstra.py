import sys; sys.path.append('../')

import numpy as np
import numba as nb
from numba.experimental import structref
import structron


__name__ = 'dijkstra'
sys.modules['dijkstra'] = sys.modules.pop('__main__')

# Heap <distance, node>

structron.register('DistHeap', globals())
DistHeap = structron.TypedHeap(DistHeapStruct, DistHeapProxy, np.float32, np.int32)

@nb.njit(cache=True)
def dijkstra_base(graph, start):
    dist = np.empty(graph.ncap, dtype=np.float32)
    seen = np.zeros(graph.ncap, dtype=np.float32)
    path = np.empty(graph.ncap, dtype=np.int32)
    npath = np.zeros(graph.ncap, dtype=np.int32)
    
    sid = graph.nmap[start]
    seen[:] = 1e10; dist[:] = -1;
    seen[sid] = 0; path[sid] = sid; npath[sid] = 1
    
    heap = DistHeap(16)
    heap.push(0, sid)
    
    eidx = graph.eidx
    while heap.size>0:
        vd = heap.top()
        v, d = vd['value'], vd['key']
        heap.pop()
        if dist[v]>=0: continue
        dist[v] = d
        for i in graph.neighbour(v, 'id'):
            cost = graph.weight(i['ebody'])         
            u = i['end']
            vu_dist = dist[v] + cost
            if vu_dist < seen[u]:
                path[u] = v
                npath[u] = npath[v] + 1
                seen[u] = vu_dist
                heap.push(vu_dist, u)
                
    return dist, path, npath

def build_dijkstra(mode='path'):
    @nb.njit(cache=True)
    def dijkstra(graph, start):
        sid = graph.nmap[start]
        dist, path, npath = dijkstra_base(graph, start)
        msk = npath > 0
        dest = graph.nidx['key'][msk]
        dist = dist[msk]
        if mode=='length':
            return dict(zip(dest, dist))

        sep = npath[msk][::-1]
        allpath = np.zeros(sep.sum(), dtype=np.int32)

        cur = 0
        for i in range(len(msk)):
            if not msk[i]: continue
            while True:
                allpath[cur] = i
                cur += 1
                if i==sid: break
                i = path[i]
        
        allpath = graph.nidx['key'][allpath[::-1]]
        allpath = np.split(allpath, np.cumsum(sep[:-1]))
        if mode=='path': return dict(zip(dest[::-1], allpath))
        
        return dict(zip(dest, dist)), dict(zip(dest[::-1], allpath))
    return dijkstra

dijkstra_length = build_dijkstra('length')
dijkstra_path = build_dijkstra('path')
dijkstra_all = build_dijkstra('both')

if __name__ == 'dijkstra':
    from time import time
    import networkx as nx
    import igraph as ig
    
    from time import time, sleep

    np.random.seed(0)
    print('generate graph with 4e5 nodes and 4e5 edges')
    n_node, n_edge = 100000, 400000
    edges = np.random.randint(0, n_node, (n_edge, 2))
    dists = np.random.rand(n_edge).astype(np.float32)
    
    # custom edge structrue
    t_edge = np.dtype([('w', np.float32)])
    
    start = time()
    
    structron.register('Graph', globals())
    Graph = structron.TypedGraph(GraphStruct, GraphProxy, np.int64, None, t_edge, weight=lambda self, x:x['w'])
    
    g = Graph(16)
    # for i in range(n_node): g.add_node(i)
    g.add_nodes(np.arange(n_node))
    g.add_edges(edges[:,0], edges[:,1], dists.view(t_edge))
    #for (u, v), d in zip(edges, dists.view(t_edge)):
    #    g.add_edge(u, v, d)
    
    print('create graph', time()-start)
    # also supportting batch mode
    # g.add_nodes(np.arange(n_node))
    # g.add_edges(edges[:,0], edges[:,1], dists.view(t_edge))
    
    # jit the distance function, and pass to the dijsktra
    

    start = time()
    myrst = dijkstra_path(g, 0)
    print('structron first cost', time()-start)
    sleep(0.1)
    
    start = time()
    myrst = dijkstra_path(g, 0)
    print('structron cost', time()-start)

    # networkx test
    g = nx.DiGraph()
    for (u, v), d in zip(edges, dists):
        g.add_edge(u, v, l=d)

    paths = {}
    sleep(0.1)
    start = time()
    nxrst = nx.single_source_dijkstra_path_length(g, 0, weight='l')
    print('networkx cost', time()-start)

    sleep(0.1)
    start = time()
    nxrst = nx.single_source_dijkstra_path(g, 0, weight='l')
    print('networkx path cost', time()-start)


    # igrahp test
    g = ig.Graph(directed=True)
    g.add_vertices(n_node)
    g.add_edges(edges)
    g.es['weight'] = dists
    sleep(0.1)
    start = time()
    igrst = g.distances(source=0, weights='weight', mode=ig.OUT)
    print('igraph cost', time() - start)


    


