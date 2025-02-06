import sys; sys.path.append('../')

import numpy as np
import numba as nb
import nbstl

# custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# TypedMemory cast Memory as dtype
PointMemory = nbstl.TypedMemory(t_point)
points = PointMemory(10)
i = points.push((1,1)) # malloc and put value
points[i] # get the value
points.pop(i) # free i
points.size, len(points) # size

# TypedDeque
PointDeque = nbstl.TypedDeque(t_point)
points = PointDeque(10)
points.push_front((1, 1)) # push front
points.push_back((2, 2)) # push back
points.first() # get the first
points.last() # get the last
points.pop_front() # pop front
points.pop_back() # pop back
points.size, len(points) # size

# TypedStack
PointStack = nbstl.TypedStack(t_point)
points = PointStack(10)
points.push((1, 1)) # push stack
points.push((2,2))
points.top() # get the top
points.pop() # pop stack
points.size, len(points) # size

# TypedQueue
PointQueue = nbstl.TypedQueue(t_point)
points = PointQueue(10)
points.push((1, 1)) # push stack
points.push((2,2))
points.top() # get the top
points.pop() # pop stack
points.size, len(points) # size

# TypedHeap
IntHeap = nbstl.TypedHeap(np.int32)
heap = IntHeap(10)
heap.push(1) # push heap
heap.top() # get the top
heap.pop() # pop heap
len(heap), heap.size # size

# TypedHash
IntHash = nbstl.TypedHash(np.int32)
hashset = IntHash(10)
hashset.push(4) # push hash
hashset.has(4) # check in
hashset.pop(4) # pop hash
len(hashset), hashset.size # size

# TypedRedBlackTree
IntTree = nbstl.TypedRBTree(np.int32)
treeset = IntTree(10)
treeset.push(1) # push tree
treeset.has(1) # check in
treeset.pop(1) # pop tree
treeset.left(1) # the key's left neighbour
treeset.right(1) # the key's right neighbour
len(treeset), treeset.size # size

# TypedAVLTree
IntTree = nbstl.TypedRBTree(np.int32)
treeset = IntTree(10)
treeset.push(1) # push tree
treeset.has(1) # check in
treeset.pop(1) # pop tree
treeset.left(1) # the key's left neighbour
treeset.right(1) # the key's right neighbour
len(treeset), treeset.size # size

# Map Mode: <Heap, Hash, RedBlackTree, AVLTree>
# here using RedBlackTree as example
IntPointTree = nbstl.TypedRBTree(np.int32, t_point) # typed it with k-v
treemap = IntPointTree(10)
treemap.push(1, (1,1)) # push key, value in the tree
treemap[1] # get the value by key
# ... other method same as TypedRedBlackTree

# Ref Mode: <All>
# here using RedBlackTree as example
PointMemory = nbstl.TypedMemory(t_point)
points = PointMemory(10)
# replace dtype with typedmemory
IntPointTree = nbstl.TypedRBTree(np.int32, PointMemory) 
treeset = IntPointTree(10, memory=points) # init with memory instance
treeset.push(1, (1,1)) # push tree
treeset.has(1) # check in
treeset.pop(1) # pop tree
treeset.left(1) # the key's left neighbour
treeset.right(1) # the key's right neighbour
len(treeset), treeset.size # size
