import sys; sys.path.append('../')

import numpy as np
import numba as nb
import nbstl

t_point = np.dtype([('x', np.float32), ('y', np.float32)])

# TypedMemory cast Memory as dtype
PointMemory = nbstl.TypedMemory(t_point)

'''
like C Memory:
push means alloc and return the index
pop means free and the space would be reuse in circle
auto expand when nedded
'''
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

'''
Deque, also could be used as Queue or Stack
push_front, push_back, pop_front, pop_back, first, last
auto expand when needed
'''
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

'''
MemoryDeque, combine a IntDeque and a TypedMemory
the IntDeque used as index (pointer)
MemoryDeque is equivalent to TypedDeque in functionality,
but maintaining pointers incurs more time overhead,
for deque need not swap, so TypedDeque is recommended.
'''
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

'''
Heap, also could be used as orderd queue
push, pop
auto expand when needed
'''
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

'''
MemoryHeap, combine a IntHeap and a TypedMemory
the IntHeap used as index (pointer)
MemoryDeque is equivalent to TypedDeque in functionality,
when The dtype is simple or the count is not large,
TypedHeap is faster than MemoryHeap.
but when the dtype is heavy and the count is large,
the MemoryHeap just swap the pointer, So may be faster.
'''
def heap_memory_test():
    print('\n>>> memory heap test:')
    PointHeap = nbstl.TypedHeap(PointMemory)
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
    
if __name__ == '__main__':
    point_memory_test()
    deque_type_test()
    deque_memory_test()
    heap_type_test()
    heap_memory_test()
