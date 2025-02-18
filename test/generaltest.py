import sys; sys.path.append('../')

import numpy as np
import numba as nb
import structron

# custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

p1, p2 = np.array([(1,1), (2,2)], t_point)

# TypedMemory cast Memory as dtype
PointMemory = structron.TypedMemory(t_point)
points = PointMemory(10)
i = points.push(p1) # malloc and put value
points[i] # get the value
points.pop(i) # free i
points.size, len(points) # size


# TypedDeque
PointDeque = structron.TypedDeque(t_point)
points = PointDeque(10)
points.push_front(p1) # push front
points.push_back(p2) # push back
points.first() # get the first
points.last() # get the last
points.pop_front() # pop front
points.pop_back() # pop back
points.size, len(points) # size


# TypedStack
PointStack = structron.TypedStack(t_point)
points = PointStack(10)
points.push(p1) # push stack
points.push(p2)
points.top() # get the top
points.pop() # pop stack
points.size, len(points) # size


# TypedQueue
PointQueue = structron.TypedQueue(t_point)
points = PointQueue(10)
points.push(p1) # push stack
points.push(p2)
points.top() # get the top
points.pop() # pop stack
points.size, len(points) # size


# TypedHeap
IntHeap = structron.TypedHeap(np.int32)
heap = IntHeap(10)
heap.push(1) # push heap
heap.top() # get the top
heap.pop() # pop heap
len(heap), heap.size # size

# TypedHash
IntHash = structron.TypedHash(np.int32)
hashset = IntHash(10)
hashset.push(4) # push hash
hashset.has(4) # check in
hashset.pop(4) # pop hash
len(hashset), hashset.size # size

# TypedRedBlackTree
IntTree = structron.TypedRBTree(np.int32)
treeset = IntTree(10)
treeset.push(1) # push tree
treeset.has(1) # check in
treeset.pop(1) # pop tree
treeset.left(1) # the key's left neighbour
treeset.right(1) # the key's right neighbour
len(treeset), treeset.size # size

# TypedAVLTree
IntTree = structron.TypedRBTree(np.int32)
treeset = IntTree(10)
treeset.push(1) # push tree
treeset.has(1) # check in
treeset.pop(1) # pop tree
treeset.left(1) # the key's left neighbour
treeset.right(1) # the key's right neighbour
len(treeset), treeset.size # size

# Map Mode: <Heap, Hash, RedBlackTree, AVLTree>
# here using RedBlackTree as example
IntPointTree = structron.TypedRBTree(np.int32, t_point) # typed it with k-v
treemap = IntPointTree(10)
treemap.push(1, p2) # push key, value in the tree
treemap[1] # get the value by key
# ... other method same as TypedRedBlackTree


# Eval Mode: <Heap, RedBlackTree, AVLTree>
eval = lambda self, p: p.x + p.y
PointHeap = structron.TypedHeap(eval, t_point) # typed it with k-v
heap = PointHeap(10)
heap.push(None, p1) # push p1 in the heap, None means evalue it
heap.push(None, p2)
heap.top() # get the top
# ... other method same as TypedHeap
