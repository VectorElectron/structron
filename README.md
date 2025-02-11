# structron 

Recently, I am trying to write some computational geometry-related functionalities. Since data structures like heaps, stacks, queues, trees, etc., are frequently used in computational geometry, these structures cannot be directly expressed as vector operations in NumPy. Also, Numba currently does not provide many advanced data structures. Therefore, I would like to develop a set of containers based on Numba to support computational geometry algorithms. structron comes from "Numba's STL container."

## Container Objects
Regarding containers:
Numba provides TypedList, which is primarily designed for general-purpose workflows and offers methods such as insert, pop, append, and more. However, when it comes to solving specific problems, there may not be efficient specialized data structures available (this is a conjecture and has not been tested). Therefore, it is suggested to implement various specialized containers for addressing specific problem domains.

## Features

1. **Seamless Integration with NumPy**:  
   All containers are implemented based on `ndarray`, ensuring seamless integration with various data processing scenarios. This makes them ideal for numerical computing and data analysis tasks.

2. **Efficient Linear Table Implementation**:  
   Linear tables (e.g., `TypedMemory`) use an array-based `next` column to replace traditional pointers. This ensures data is always stored in order, and released slots are efficiently reused in a circular manner.

3. **Pure Array-Based Red-Black Tree**:  
   The Red-Black Tree is implemented purely using arrays, with `left` and `right` columns recording the indices of corresponding nodes in the array. This design eliminates the need for dynamic memory allocation and pointer management, improving performance and memory efficiency.

4. **Outstanding Performance**:  
   Extensive testing shows that the performance of `Hash` and `RedBlackTree` is comparable to C++'s STL. This makes the containers suitable for high-performance computing and real-time applications.

### Why Choose These Containers?
- **Memory Efficiency**: Leveraging `ndarray` and array-based designs minimizes memory overhead.
- **High Performance**: Optimized implementations ensure competitive performance with low-level languages.
- **Flexibility**: Supports both standalone and shared memory modes, catering to diverse use cases.
- **Ease of Use**: Designed to work seamlessly with Python's scientific computing ecosystem.

## `TypedMemory` Container

`TypedMemory` is a high-performance memory management container for custom data types, built with `numba`. It dynamically manages memory allocation and deallocation, maintaining a free list for efficient reuse of released slots and automatically expanding capacity when necessary.

### Example Code

```python
import numpy as np
import numba as nb
import structron

# Custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# Define a TypedMemory class for the custom dtype
PointMemory = structron.TypedMemory(t_point)

# Instantiate a memory pool with a capacity of 10
points = PointMemory(10)

# Allocate memory and store a value
i = points.push((1, 1))  # Returns the index of the allocated slot

# Access the value at index `i`
points[i]  # Returns (x=1.0, y=1.0)

# Deallocate the memory at index `i`
points.pop(i)  # Frees the slot for reuse

# Get container size
points.size  # Current number of allocated elements
len(points)  # Total capacity of the memory pool
```

### Methods

- `TypedMemory(dtype)`: Defines a memory pool class for the specified `dtype`.
- `push(value)`: Allocates a slot and stores `value`. Returns the index.
- `pop(index)`: Deallocates the slot at `index` and adds it to the free list.
- `__getitem__(index)`: Retrieves the value at `index`.
- `size`: Returns the current number of allocated slots.


## `TypedDeque` Container

`TypedDeque` is a double-ended queue (deque) implementation for custom data types, optimized with `numba`. It allows efficient insertion and removal of elements at both ends.

### Example Code

```python
import numpy as np
import numba as nb
import structron

# Custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# Define a TypedDeque class for the custom dtype
PointDeque = structron.TypedDeque(t_point)

# Instantiate a deque with a capacity of 10
points = PointDeque(10)

# Push elements
points.push_front((1, 1))  # Push to the front
points.push_back((2, 2))   # Push to the back

# Access elements
points.first()  # Get the first element
points.last()   # Get the last element

# Pop elements
points.pop_front()  # Remove the first element
points.pop_back()   # Remove the last element

# Get container size
points.size  # Current number of elements
len(points)  # Total capacity of the deque
```

### Methods

- `TypedDeque(dtype)`: Defines a deque class for the specified `dtype`.
- `push_front(value)`: Adds `value` to the front of the deque.
- `push_back(value)`: Adds `value` to the back of the deque.
- `first()`: Returns the first element in the deque.
- `last()`: Returns the last element in the deque.
- `pop_front()`: Removes and returns the first element.
- `pop_back()`: Removes and returns the last element.
- `size`: Returns the current number of elements in the deque.


## `TypedStack` Container

`TypedStack` is a stack implementation for custom data types, optimized with `numba`. It follows the Last-In-First-Out (LIFO) principle.

### Example Code

```python
import numpy as np
import numba as nb
import structron

# Custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# Define a TypedStack class for the custom dtype
PointStack = structron.TypedStack(t_point)

# Instantiate a stack with a capacity of 10
points = PointStack(10)

# Push elements
points.push((1, 1))  # Push to the stack
points.push((2, 2))  # Push another element

# Access the top element
points.top()  # Returns (x=2.0, y=2.0)

# Pop the top element
points.pop()  # Removes and returns (x=2.0, y=2.0)

# Get container size
points.size  # Current number of elements in the stack
len(points)  # Same as `size`, not the total capacity
```

### Methods

- `TypedStack(dtype)`: Defines a stack class for the specified `dtype`.
- `push(value)`: Adds `value` to the top of the stack.
- `top()`: Returns the top element without removing it.
- `pop()`: Removes and returns the top element.
- `size`: Returns the current number of elements in the stack.


## `TypedQueue` Container

`TypedQueue` is a queue implementation for custom data types, optimized with `numba`. It follows the First-In-First-Out (FIFO) principle.

### Example Code

```python
import numpy as np
import numba as nb
import structron

# Custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# Define a TypedQueue class for the custom dtype
PointQueue = structron.TypedQueue(t_point)

# Instantiate a queue with a capacity of 10
points = PointQueue(10)

# Push elements
points.push((1, 1))  # Add to the queue
points.push((2, 2))  # Add another element

# Access the front element
points.top()  # Returns (x=1.0, y=1.0)

# Pop the front element
points.pop()  # Removes and returns (x=1.0, y=1.0)

# Get container size
points.size  # Current number of elements in the queue
```

### Methods

- `TypedQueue(dtype)`: Defines a queue class for the specified `dtype`.
- `push(value)`: Adds `value` to the back of the queue.
- `top()`: Returns the front element without removing it.
- `pop()`: Removes and returns the front element.
- `size`: Returns the current number of elements in the queue.


## `TypedHeap` Container

`TypedHeap` is a heap (priority queue) implementation for custom data types, optimized with `numba`. It maintains elements in a way that allows efficient access to the smallest (or largest) element.

### Example Code

```python
import numpy as np
import numba as nb
import structron

# Define a TypedHeap class for int32
IntHeap = structron.TypedHeap(np.int32)

# Instantiate a heap with a capacity of 10
heap = IntHeap(10)

# Push elements
heap.push(1)  # Add to the heap
heap.push(2)  # Add another element

# Access the top element
heap.top()  # Returns the smallest element (1)

# Pop the top element
heap.pop()  # Removes and returns the smallest element (1)

# Get container size
heap.size  # Current number of elements in the heap
len(heap)  # Same as `size`
```

### Methods

- `TypedHeap(dtype)`: Defines a heap class for the specified `dtype`.
- `push(value)`: Adds `value` to the heap.
- `top()`: Returns the smallest element without removing it.
- `pop()`: Removes and returns the smallest element.
- `size`: Returns the current number of elements in the heap.

## `TypedHash` Container

`TypedHash` is a hash-based container for custom data types, optimized with `numba`. It provides efficient insertion, deletion, and lookup operations.

### Example Code

```python
import numpy as np
import numba as nb
import structron

# Define a TypedHash class for int32
IntHash = structron.TypedHash(np.int32)

# Instantiate a hash container with a capacity of 10
hashset = IntHash(10)

# Insert an element
hashset.push(4)  # Add to the hash container

# Check if an element exists
hashset.has(4)  # Returns True if the element is present

# Remove an element
hashset.pop(4)  # Removes the element from the container

# Get container size
hashset.size  # Current number of elements in the container
len(hashset)  # Same as `size`
```

### Methods

- `TypedHash(dtype)`: Defines a hash container class for the specified `dtype`.
- `push(value)`: Adds `value` to the hash container.
- `has(value)`: Checks if `value` exists in the container.
- `pop(value)`: Removes `value` from the container.
- `size`: Returns the current number of elements in the container.

## `TypedRedBlackTree` Container

`TypedRedBlackTree` is a Red-Black Tree implementation for custom data types, optimized with `numba`. It provides efficient insertion, deletion, lookup, and neighbor access operations.

### Example Code

```python
import numpy as np
import numba as nb
import structron

# Define a TypedRedBlackTree class for int32
IntTree = structron.TypedRBTree(np.int32)

# Instantiate a tree with a capacity of 10
treeset = IntTree(10)

# Insert an element
treeset.push(1)  # Add to the tree

# Check if an element exists
treeset.has(1)  # Returns True if the element is present

# Remove an element
treeset.pop(1)  # Removes the element from the tree

# Access neighbors
treeset.left(1)  # Returns the left neighbor of the key
treeset.right(1) # Returns the right neighbor of the key

# Get container size
treeset.size  # Current number of elements in the tree
len(treeset)  # Same as `size`
```

### Methods

- `TypedRBTree(dtype)`: Defines a Red-Black Tree class for the specified `dtype`.
- `push(value)`: Adds `value` to the tree.
- `has(value)`: Checks if `value` exists in the tree.
- `pop(value)`: Removes `value` from the tree.
- `left(value)`: Returns the left neighbor of `value` (smaller than `value`).
- `right(value)`: Returns the right neighbor of `value` (larger than `value`).
- `size`: Returns the current number of elements in the tree.


## `TypedAVLTree` Container

`TypedAVLTree` is an AVL Tree implementation for custom data types, optimized with `numba`. Its interface is identical to `TypedRedBlackTree`.

### Example Code

```python
import numpy as np
import numba as nb
import structron

# Define a TypedAVLTree class for int32
IntTree = structron.TypedAVLTree(np.int32)

# Same as TypedRBTree
```

### Methods

The interface is identical to `TypedRedBlackTree`. Refer to the `TypedRedBlackTree` documentation for details.


## Map Mode: `<Heap, Hash, RedBlackTree, AVLTree>`

The map mode allows using `Heap`, `Hash`, `RedBlackTree`, and `AVLTree` as key-value containers. The `TypedXXX` function accepts two parameters: the key type and the value type. The interface extends the non-map mode by requiring a key and value for `push`, and supports accessing values using the key as a subscript.

### Example Code (Using `RedBlackTree`)

```python
import numpy as np
import numba as nb
import structron

# Custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# Define a TypedRBTree class for int32 keys and point values
IntPointTree = structron.TypedRBTree(np.int32, t_point)

# Instantiate a tree with a capacity of 10
treemap = IntPointTree(10)

# Insert a key-value pair
treemap.push(1, (1, 1))  # Push key=1, value=(1, 1)

# Access the value by key
treemap[1]  # Returns the value (x=1.0, y=1.0)

# Other methods are the same as the non-map mode
# e.g., treemap.has(1), treemap.pop(1), treemap.left(1), treemap.right(1)
```

### Supported Containers

The map mode is supported for the following containers:
- **Heap**: `TypedHeap(key_dtype, value_dtype)`
- **Hash**: `TypedHash(key_dtype, value_dtype)`
- **RedBlackTree**: `TypedRBTree(key_dtype, value_dtype)`
- **AVLTree**: `TypedAVLTree(key_dtype, value_dtype)`

### Key Differences from Non-Map Mode
1. The `TypedXXX` function accepts two parameters: `key_dtype` and `value_dtype`.
2. The `push` method requires both a key and a value: `push(key, value)`.
3. Values can be accessed using the key as a subscript: `container[key]`.
4. Other methods (e.g., `has`, `pop`, `left`, `right`) remain the same as in the non-map mode.

## Ref Mode: `<All>`

Ref Mode allows containers to store references to data in a shared `TypedMemory` instance. This is useful when working with large or complex data types, or when multiple containers need to share the same memory pool. In this mode, the container only stores integer references, while the actual data resides in the `TypedMemory`.

### Example Code (Using `RedBlackTree`)

```python
import numpy as np
import numba as nb
import structron

# Custom point structure
t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# Create a TypedMemory instance
PointMemory = structron.TypedMemory(t_point)
points = PointMemory(10)

# Define a TypedRBTree class for int32 keys and PointMemory references
IntPointTree = structron.TypedRBTree(np.int32, PointMemory)

# Instantiate the tree with the shared memory instance
treeset = IntPointTree(10, memory=points)

# Insert a key-value pair
treeset.push(1, (1, 1))  # Push key=1, value=(1, 1) into the shared memory

# Check if a key exists
treeset.has(1)  # Returns True if the key is present

# Remove a key-value pair
treeset.pop(1)  # Removes the key and its associated value from the shared memory

# Access neighbors
treeset.left(1)  # Returns the left neighbor of the key
treeset.right(1) # Returns the right neighbor of the key

# Get container size
treeset.size  # Current number of elements in the tree
len(treeset)  # Same as `size`
```

### Key Features of Ref Mode
1. **Shared Memory**: Multiple containers can share the same `TypedMemory` instance.
2. **Efficient Storage**: Containers only store integer references, reducing memory overhead.
3. **Complex Data Types**: Ideal for large or complex data structures where direct storage is inefficient.
4. **Multi-Container Collaboration**: Enables multiple containers to work on the same data pool.

### Usage Notes
- Replace the `dtype` parameter in `TypedXXX` with a `TypedMemory` instance.
- Pass the `TypedMemory` instance to the container during initialization using the `memory` parameter.
- The `push` method stores the value in the shared `TypedMemory` and associates it with the key.
- The `pop` method removes the key and its associated value from the shared memory.
- Other methods (e.g., `has`, `left`, `right`) work as in the non-ref mode.

### Supported Containers
Ref Mode is supported for all containers:
- **Heap**: `TypedHeap(key_dtype, TypedMemory)`
- **Hash**: `TypedHash(key_dtype, TypedMemory)`
- **RedBlackTree**: `TypedRBTree(key_dtype, TypedMemory)`
- **AVLTree**: `TypedAVLTree(key_dtype, TypedMemory)`


### More Container
More containers will be implemented successively. Welcome to provide suggestions or participate in project development.

## Demo
The original intention of developing structron was to create a framework that facilitates the implementation of computational geometry algorithms. Here, we will demonstrate the usage of structron using a simple yet classic example, and perform a performance test.

### Convex Hull
Algorithm Implementation:
1. Sort the points by their X-coordinate.
2. Build the upper half: Start from the leftmost point and push it onto the stack. For each new point, check the last two points on the stack. If they form a right turn with the new point, push the new point onto the stack. Otherwise, pop the top element from the stack until a right turn is obtained.
3. Build the lower half: Start from the rightmost point and push it onto the stack. For each new point, check the last two points on the stack. If they form a right turn with the new point, push the new point onto the stack. Otherwise, pop the top element from the stack until a right turn is obtained.
4. Combine the upper and lower halves to obtain the convex hull.

```python
import numpy as np
import numba as nb
import structron

# build Point dtype and PointStack
t_point = np.dtype([('x', np.float32), ('y', np.float32)])
PointStack = structron.TypedStack(t_point)

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
        hull.push((p2.x, p2.y))
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
![1714464150709](https://github.com/Image-Py/structron/assets/24822467/576eec48-5d0d-4d17-a84d-58ca70279845)

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
