import sys; sys.path.append('../')
import matplotlib.pyplot as plt

import numpy as np
import numba as nb
import nbstl

t_point = np.dtype([('x', np.float32), ('y', np.float32)])
PointStack = nbstl.TypedStack(t_point)

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

@nb.njit
def convexhull(pts):
    idx = np.argsort(pts['x'])
    up = convex_line(pts, idx)
    down = convex_line(pts, idx[::-1])
    return np.concatenate((up, down[1:]))

if __name__ == '__main__':
    from time import time

    pts = np.random.randn(102400,2).astype(np.float32)
    pts = pts.ravel().view(t_point)

    hull = convexhull(pts)
    start = time()
    hull = convexhull(pts)
    print('convex hull of 102400 point cost:', time()-start)

    from shapely import geometry as geom
    mpoints = geom.MultiPoint(pts.view(np.float32).reshape(-1,2))
    start = time()
    cvxh = mpoints.convex_hull
    print('the same points by shapely cost:', time()-start)
    
    plt.plot(pts['x'], pts['y'], 'r.')
    plt.plot(hull['x'], hull['y'], 'g-')
    plt.title('Convex Hull')
    plt.show()
