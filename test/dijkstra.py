import sys; sys.path.append('../')

import numpy as np
import numba as nb
from numba.experimental import structref
import structron


__name__ = 'dijkstra'
sys.modules['dijkstra'] = sys.modules.pop('__main__')

# Heap <distance, node>
class DistHeapStruct(nb.types.StructRef): pass
class DistHeapProxy(structref.StructRefProxy): pass
DistHeap = structron.TypedHeap(DistHeapStruct, DistHeapProxy, np.float32, np.int32)

@nb.njit(cache=True)
def _dijkstra(graph, start):
    dist = np.empty(graph.ncap, dtype=np.float32)
    seen = np.zeros(graph.ncap, dtype=np.float32)
    sid = graph.nmap[start]
    seen[:] = 1e10; dist[:] = -1; seen[sid] = 0
    
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
                seen[u] = vu_dist
                heap.push(vu_dist, u)
    
    msk = dist > -1
    dest = graph.nidx[msk]['key']
    dist = dist[msk]
    return dict(zip(dest, dist))


#def dijkstra(graph, start, f=nofunc):
#    return _dijkstra(graph, start, f)
    # return dict(rst)

if __name__ == 'dijkstra':
    from time import time
    import networkx as nx
    import igraph as ig
    
    from time import time

    np.random.seed(0)
    print('generate graph with 4e5 nodes and 4e5 edges')
    n_node, n_edge = 100000, 400000
    edges = np.random.randint(0, n_node, (n_edge, 2))
    dists = np.random.rand(n_edge).astype(np.float32)
    
    # custom edge structrue
    t_edge = np.dtype([('w', np.float32)])
    
    start = time()
    
    class GraphStruct(nb.types.StructRef): pass
    class GraphProxy(structref.StructRefProxy): pass
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
    myrst = _dijkstra(g, 0)
    print('structron first cost', time()-start)
    
    start = time()
    myrst = _dijkstra(g, 0)
    print('structron cost', time()-start)

    # networkx test
    g = nx.DiGraph()
    for (u, v), d in zip(edges, dists):
        g.add_edge(u, v, l=d)

    paths = {}
    start = time()
    nxrst = nx.single_source_dijkstra_path_length(g, 0, weight='l')
    print('networkx cost', time()-start)



    # igrahp test
    g = ig.Graph(directed=True)
    g.add_vertices(n_node)
    g.add_edges(edges)
    g.es['weight'] = dists

    start = time()
    igrst = g.distances(source=0, weights='weight', mode=ig.OUT)
    print('igraph cost', time() - start)


    


