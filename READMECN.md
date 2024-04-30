# nbstl [English](README.md) | [中文](READMECN.md)  
最近在试图编写一些计算几何相关的功能，由于计算几何中频繁的用到堆，栈，队列，树等数据结构，这些逻辑无法直接表达成Numpy的向量运算，而Numba中暂时没有提供太多的高级数据结构，因而希望基于numba编写的一套容器，以支持计算几何算法，名称由来是numba’s STL container。

## 容器对象
关于容器：
numba中提供了TypedList，然而TypedList主要是面向通用流程的，提供了insert，pop，append等多种方法。然而在解决一些特定问题时候，并没有专用数据结构高效。（猜想，并未实测）一下实现各种专用容器。

### Memory：
实现原理: 底层基于数组实现，内置idx和body。其中idx指向下一个位置，初始状态下是顺序递增的，并记录tail，中途释放的，将tail只想释放位置，并将释放位置作为新的tail，从而实现空间滚动利用，当容量不足时会自动扩张。

成员:
* `idx`: 维护索引，指向下一个空闲位置。
* `body`: 数据数组
* `size`: 数量
* `cap`: 容量

方法：
* `push`: 添加对象，返回索引，可以理解为alloc。
* `pop`: 弹出指定位置的对象，同时将位置接到末尾，可以理解为free。

### TypedMemory
为Memory的具体类型的JIT实现，下面以Point举例应用，首先用numpy定义类型Point，然后用TypedMemory函数，得到具体类型PointMemory，再实例化PointMemory，由于定义了具体类型，push可以传入x, y两个参数，pop也将返回(x, y)的record。
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
实现原理: 底层基于数组实现，内置body，通过维护head，tail，可以向前后追加数据。当容量不足时自动填充。

成员:
* `body`: 数据数组
* `size`: 数量
* `cap`: 容量

方法：
* `push_front`: 向头部添加数据
* `push_back`: 向尾部添加数据
* `pop_front`: 从头部弹出
* `pop_back`: 从尾部弹出
* `first(idx=0)`: 获得头部第n元素
* `last(idx=0)`: 获得尾部第n元素

### TypedDeque
为Memory的具体类型的JIT实现，下面以Point举例应用，首先用numpy定义类型Point，然后用TypedDeque函数，得到具体类型PointDeque，再实例化PointDeque，由于定义了具体类型，push_front/back可以传入x, y两个参数，pop_front/back也将返回(x, y)的record。
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
MemoryDeque 是TypedDeque(uint32)与TypedMemory的组装类，是用IntDeque维护索引，同时将对象实体放到Memory上，功能和接口上等同于TypedDeque。创建类型时，需使用TypedMemory作为参数，而非dtype。由于多了一层索引关系，因而性能略低于TypedDeque，并且Deque不存在大规模的元素移动操作，因而通常情况推荐用TypedDeque。
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
TypedDeque的继承类，`push_back` 重定义为 `push`，`pop_back` 重定义成 `pop`，`last` 重定义为 `top`。使得操作行为更像Stack，同时Deque的原有方法依然保留。

### TypedQueue
TypedDeque的继承类，`push_back` 重定义为 `push`，`pop_front` 重定义成 `pop`，`first` 重定义为 `top`。使得操作行为更像Queue，同时Deque的原有方法依然保留。

### Heap
实现原理: 底层基于数组实现，内置idx, body，idx为float32类型的key，用于排序，body用于存放对象。

成员:
* `idx`: 排序主键
* `body`: 数据数组
* `size`: 数量
* `cap`: 容量

方法：
* `push`: 像堆中添加元素
* `pop`: 弹出元素
  
### TypedHeap
为Heap的具体类型的JIT实现，下面以Point举例应用，首先用numpy定义类型Point，然后用TypedHeap函数，得到具体类型PointHeap，再实例化PointHeap，由于定义了具体类型，push可以传入x, y两个参数，pop也将返回(x, y)的record。
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
MemoryHeap 是TypedHeap(uint32)与TypedMemory的组装类，是用IntHeap维护索引，同时将对象实体放到Memory上，功能和接口上等同于TypedHeap。创建类型时，需使用TypedMemory作为参数，而非dtype。由于多了一层索引关系，因而略微增加了访问开销，但由于堆需要动态维护次序，当数据量较大时，push和pop操作将涉及较多次数的swap，又当数据结构较为复杂的时候，swap操作将拷贝较多的内存。而MemoryHeap只维护索引，因此在数据量大，数据结构重的时候MemoryHeap将更具优势。

### More Container
更多容器将陆续实现，欢迎各位提供意见，或参与项目编写。

## 用法示例
编写nbstl的初衷时为了构建一套容器，便于计算几何算法的实现。这里以一个简单而经典的例子，演示nbstl的用法，并进行性能测试。

### 1. 凸包
算法实现：
1. 按照X排序
2. 构建上半边：从左到右压栈，新点和栈顶两点为右转关系，则push，否则pop栈顶元素
3. 构建下半边：从右到左压栈，新点和栈顶两点为右转关系，则push，否则pop栈顶元素
4. 合并上下半边，得到凸包

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

然后我们在相同规模的数据上，使用shapely进行凸包计算，做性能比对：
```python
from shapely import geometry as geom

pts = np.random.randn(102400, 2).astype(np.float32)
mpoints = geom.MultiPoint(pts)
start = time()
cvxh = mpoints.convex_hull
print('the same points by shapely cost:', time()-start)
```
我们获得如下结果：
```
convex hull of 102400 point cost: 0.01891
the same points by shapely cost: 0.04986

convex hull of 1024000 point cost: 0.23035
the same points by shapely cost: 1.08539
```
随着数据量增加，已经明显快于shapely了，注意，这里排除了数据构建上花费的时间，只是单纯的比对凸包生成的算法。事实上，大数据量从numpy到shapely的io开销也是巨大的。所以我认为用python，基于numba构建一些高效的2d, 3d的几何算法是非常有意义的。
## 招募
此项工作涉及较多，较深入的numba使用技巧，也涉及到专业的计算几何知识，欢迎感兴趣的朋友们，共同参与开发。