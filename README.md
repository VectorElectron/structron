# nbstl [English](README.md) | [中文](READMECN.md)  

Recently, I am trying to write some computational geometry-related functionalities. Since data structures like heaps, stacks, queues, trees, etc., are frequently used in computational geometry, these structures cannot be directly expressed as vector operations in NumPy. Also, Numba currently does not provide many advanced data structures. Therefore, I would like to develop a set of containers based on Numba to support computational geometry algorithms. nbstl comes from "Numba's STL container."

## Container Objects
Regarding containers:
Numba provides TypedList, which is primarily designed for general-purpose workflows and offers methods such as insert, pop, append, and more. However, when it comes to solving specific problems, there may not be efficient specialized data structures available (this is a conjecture and has not been tested). Therefore, it is suggested to implement various specialized containers for addressing specific problem domains.

### Memory：
Implementation principle: The underlying implementation is based on an array, with built-in "idx" and "body" components. The "idx" points to the next available position, initially set to increment sequentially, and it also keeps track of the "tail" position. When an element is released during the process, the "tail" is updated to the released position, allowing for efficient utilization of space through rolling. The data structure automatically expands when capacity is insufficient.

Fidlds:
* `idx`: This maintains the index pointing to the next available position.
* `body`: This is the data array.
* `size`: This represents the current number of elements in the container.
* `cap`: This indicates the capacity of the container.

Methods:
* `push`: This method adds an object to the container and returns the corresponding index. It can be understood as an "alloc" operation.
* `pop`: This method removes the object at the specified position and places it at the end of the container. It can be understood as a "free" operation.
  
### TypedMemory
the specific type JIT implementation of Memory, let's take the example of Point. First, define the Point type using numpy. Then, use the TypedMemory function to obtain the specific type PointMemory. Next, instantiate PointMemory. Since a specific type is defined, the push function can take x and y as parameters, and pop will also return a record (x, y).
```python
import numpy as np
import numba as nb
import nbstl

t_point = np.dtype([('x', np.float32), ('y', np.float32)])

PointMemory = nbstl.TypedMemory(t_point)

def point_memory_test():
    print('\n>>> point memory test:')
    points = PointMemory(10)
    i1 = points.push(1, 2)
    i2 = points.push(3, 4)

    p = points[i1]
    print('point1:', p)
    print('point.xy:', p.x, p.y)

    print('size:', points.size)
    print('pop i2:', points.pop(i2))
    print('size:', points.size)
```
### Deque：
Implementation principle: It is based on an array at the underlying level, has a built-in body. By maintaining the head and tail, data can be appended both forward and backward. It automatically fills in when the capacity is insufficient.

Fields:
* `body`: struct array
* `size`: count
* `cap`: capacity

Methods：
* `push_front`: push front
* `push_back`: push back
* `pop_front`: pop front
* `pop_back`: pop back
* `first(idx=0)`: get the n first element
* `last(idx=0)`: get the n last element

### TypedDeque
the specific type JIT implementation of Deque, let's take the example of a Point. First, we define the Point type using numpy. Then, we use the TypedDeque function to obtain the specific type called PointDeque. Next, we instantiate PointDeque, and since we have defined the specific type, we can pass x and y as parameters to the push_front/back methods. Similarly, the pop_front/back methods will return a record containing (x, y).
```python
import numpy as np
import numba as nb
import nbstl

t_point = np.dtype([('x', np.float32), ('y', np.float32)])

def deque_type_test():
    print('\n>>> point deque test:')
    PointDeque = nbstl.TypedDeque(t_point)
    points = PointDeque(10)
    print('push 3 points')
    points.push_front(1, 1)
    points.push_back(2, 2)
    points.push_front(0, 0)
    print('size:', points.size)
    print('first point:', points.first())
    print('first point:', points.last())
    print('pop front:', points.pop_front())
    print('pop back:', points.pop_back())
    print('size:', points.size)
```

### MemoryDeque
MemoryDeque is an JIT class that combines TypedDeque(uint32) and TypedMemory. It uses IntDeque to maintain indices and stores object entities in memory. It has the same functionality and interface as TypedDeque. When creating a type, TypedMemory should be used as a parameter instead of dtype. Due to the additional layer of index relationships, its performance is slightly lower than TypedDeque. Additionally, Deque does not support large-scale element movement operations, so TypedDeque is generally recommended in typical scenarios.
```python
import numpy as np
import numba as nb
import nbstl

t_point = np.dtype([('x', np.float32), ('y', np.float32)])
PointMemory = nbstl.TypedMemory(t_point)

def deque_memory_test():
    print('\n>>> memory deque test:')
    PointDeque = nbstl.TypedDeque(PointMemory)
    points = PointDeque(10)
    print('push 3 points')
    points.push_front(1, 1)
    points.push_back(2, 2)
    points.push_front(0, 0)
    print('size:', points.size)
    print('first point:', points.first())
    print('first point:', points.last())
    print('pop front:', points.pop_front())
    print('pop back:', points.pop_back())
    print('size:', points.size)
```

### TypedStack
extended from TypedDeque，`push_back` rename as `push`，`pop_back` rename as `pop`，`last` rename as `top` Making the operational behavior more like a Stack while retaining the original methods of Deque.

### TypedQueue
extended from TypedDeque，`push_back` rename as `push`，`pop_front` rename as `pop`，`first` rename as `top` Making the operational behavior more like a Queue while retaining the original methods of Deque.

### Heap
Implementation principle: The underlying implementation is based on an array, with built-in `idx` and `body` properties. `idx` is a float32 type key used for sorting, while `body` is used to store objects.

Fields:
* idx: key
* body: struct array
* size: count
* cap: capacity

Methods:
* push: push to heat
* pop: pop from heat
  
### TypedHeap
the specific type JIT implementation of Heap, taking Point as an example, we first define the Point type using numpy. Then, we use the TypedHeap function to obtain the specific type PointHeap. Finally, we instantiate PointHeap. Since we have defined the specific type, we can pass both x and y parameters to push, and pop will return a record of (x, y).
```python
import numpy as np
import numba as nb
import nbstl

t_point = np.dtype([('x', np.float32), ('y', np.float32)])

def heap_type_test():
    print('\n>>> point heap test:')
    PointHeap = nbstl.TypedHeap(t_point)
    points = PointHeap(10)
    print('push key:0.5, value:(5,5)')
    points.push(0.5, 5, 5)
    print('push key:0.2, value:(2,2)')
    points.push(0.2, 2, 2)
    print('push key:0.7, value:(7,7)')
    points.push(0.7, 7, 7)
    print('size:', points.size)
    print('pop min:', points.pop())
    print('pop min:', points.pop())
    print('size:', points.size)
```

### MemoryHeap
MemoryHeap is an JIT class that combines TypedHeap(uint32) and TypedMemory. It functions and interfaces in the same way as TypedHeap. When creating a type, TypedMemory is used as a parameter instead of dtype. Due to the additional layer of indexing, there is a slight increase in access overhead. However, since the heap needs to dynamically maintain the order, when the data size is large, push and pop operations will involve a significant number of swaps. Additionally, when the data structure is heavy, swap operations will involve copying a large amount of memory. On the other hand, MemoryHeap only maintains the index, so it has an advantage in terms of performance when dealing with large data sizes and heavy data structures.

### More Container
More containers will be implemented successively. Welcome to provide suggestions or participate in project development.

## Demo
The original intention of developing nbstl was to create a framework that facilitates the implementation of computational geometry algorithms. Here, we will demonstrate the usage of nbstl using a simple yet classic example, and perform a performance test.

### Convex Hull
Algorithm Implementation:
1. Sort the points by their X-coordinate.
2. Build the upper half: Start from the leftmost point and push it onto the stack. For each new point, check the last two points on the stack. If they form a right turn with the new point, push the new point onto the stack. Otherwise, pop the top element from the stack until a right turn is obtained.
3. Build the lower half: Start from the rightmost point and push it onto the stack. For each new point, check the last two points on the stack. If they form a right turn with the new point, push the new point onto the stack. Otherwise, pop the top element from the stack until a right turn is obtained.
4. Combine the upper and lower halves to obtain the convex hull.

```python
import numpy as np
import numba as nb
import nbstl

# build Point dtype and PointStack
t_point = np.dtype([('x', np.float32), ('y', np.float32)])
PointStack = nbstl.TypedStack(t_point)

# push to stack one by one, if not turn right, pop
@nb.njit
def convex_line(pts, idx):
    hull = PointStack(128)
    for i in idx:
        p2 = pts[i]
        while hull.size>1:
            p1 = hull.top(0)
            p0 = hull.top(1)
            s = p0.x*p1.y - p0.y*p1.x
            s += p1.x*p2.y - p1.y*p2.x
            s += p2.x*p0.y - p2.y*p0.x
            if s<-1e-6: break
            hull.pop()
        hull.push(p2.x, p2.y)
    return hull.body[:hull.size]

# get up line and down line, then concat the hull
@nb.njit
def convexhull(pts):
    idx = np.argsort(pts['x'])
    up = convex_line(pts, idx)
    down = convex_line(pts, idx[::-1])
    return np.concatenate((up, down[1:]))

if __name__ == '__main__':
    from time import time
    # use randn to make 102400 random points
    pts = np.random.randn(102400, 2).astype(np.float32)
    pts = pts.ravel().view(t_point)

    hull = convexhull(pts)
    start = time()
    hull = convexhull(pts)
    print('convex hull of 102400 point cost:', time()-start)
```
![1714464150709](https://github.com/Image-Py/nbstl/assets/24822467/576eec48-5d0d-4d17-a84d-58ca70279845)

Then, we perform performance comparison using Shapely for convex hull computation on datasets of the same.
```python
from shapely import geometry as geom

pts = np.random.randn(102400, 2).astype(np.float32)
mpoints = geom.MultiPoint(pts)
start = time()
cvxh = mpoints.convex_hull
print('the same points by shapely cost:', time()-start)
```
we got the result below:
```
convex hull of 102400 point cost: 0.01891
the same points by shapely cost: 0.04986

convex hull of 1024000 point cost: 0.23035
the same points by shapely cost: 1.08539
```
With the increase in data volume, it has become evident that the performance of shapely is slower. Please note that this comparison is solely focused on the generation of convex hulls, excluding the time spent on data construction. In fact, the I/O overhead from numpy to shapely for large data sets is also significant. Therefore, I believe it is meaningful to build efficient 2D and 3D geometry algorithms using Python and leveraging numba.

## Recruitment
This project involves extensive and in-depth knowledge of numba and requires expertise in computational geometry. We welcome interested individuals to join us in the development process.
