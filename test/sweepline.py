import sys; sys.path.append('../')

import numpy as np
import numba as nb

from structron import TypedAVLTree
from structron import TypedMemory

t_point = np.dtype([('x', np.float32), ('y', np.float32)])
t_line = np.dtype([('x1', np.float32), ('y1', np.float32),
                   ('x2', np.float32), ('y2', np.float32)])

t_event = np.dtype([('tp', np.uint8), ('s', np.int32), ('e', np.int32), ('y', np.float32)])

@nb.experimental.jitclass(
    [('p', nb.float32[:]), ('ip', nb.uint32[:]), ('lp', nb.uint64[:])])
class EvalPoint:
    def __init__(self):
        self.p = np.array([0,0], dtype=np.float32)
    
    def eval(self, x, y):
        ip = self.p.view(np.uint32)
        lp = self.p.view(np.uint64)
        self.p[0] = x
        self.p[1] = -y
        ix, iy = ip[0], ip[1]
        if ix & (1 << 31): ip[0] = ~ix
        else:  ip[0] = ix | (1 << 31)
        if iy & (1 << 31): ip[1] = ~iy
        else:  ip[1] = iy | (1 << 31)
        return lp[0]>>1
        
@nb.njit
def eval_p(x, y):
    # return EvalPoint().eval(x, y)
    return y * -1e4 + x * 1e-4

@nb.njit
def cross(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10: return (np.nan, np.nan)
    numerator_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    numerator_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    t = numerator_t / denom
    u = -numerator_u / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return (np.nan, np.nan)

def eval_x(self, p):
    if p.y1==p.y2: return p.x1
    k = (self.sweepy - p.y1) / (p.y2 - p.y1)
    return p.x1 + k * (p.x2 - p.x1) 

ActiveLine = TypedAVLTree(eval_x, t_line, {'sweepy':np.float32})
EventQueue = TypedAVLTree(np.float64, t_event)
PointMemory = TypedMemory(t_point)

@nb.njit
def init_events(lines):
    e0 = np.zeros(1, t_event)[0]
    l0 = np.zeros(1, t_line)[0]
    pts = lines.ravel()
    events = EventQueue(128)
    
    for i in range(0, len(pts), 2):
        p1, p2 = pts[i], pts[i+1]
        v1, v2 = eval_p(p1['x'], p1['y']), eval_p(p2['x'], p2['y'])
        dir = v1 < v2
        e0['s'] = i if dir else i+1
        e0['e'] = i+1 if dir else i
        e0['tp'] = 0
        e0['y'] = p1['y'] if dir else p2['y']
        # print(pts[e0['s']], pts[e0['e']], e0)
        events.push(min(v1,v2), e0)
        e0['tp'] = 1
        e0['y'] = p2['y'] if dir else p1['y']
        # print(pts[e0['s']], pts[e0['e']], e0)
        events.push(max(v1,v2), e0)
    return events

'''
@nb.njit
def init_events(lines):
    e0 = np.zeros(1, t_event)[0]
    l0 = np.zeros(1, t_line)[0]
    pts = lines.ravel()

    events = np.zeros(len(pts), dtype=t_event)
    weight = np.zeros(len(pts), dtype=np.float64)
    
    for i in range(0, len(pts), 2):
        p1, p2 = pts[i], pts[i+1]
        v1, v2 = eval_p(p1['x'], p1['y']), eval_p(p2['x'], p2['y'])
        dir = v1 < v2
        e0['s'] = i if dir else i+1
        e0['e'] = i+1 if dir else i
        e0['tp'] = 0
        e0['y'] = p1['y'] if dir else p2['y']
        events[i] = e0; weight[i] = min(v1,v2)
        e0['tp'] = 1
        e0['y'] = p2['y'] if dir else p1['y']
        events[i+1] = e0; weight[i+1] = max(v1,v2)
    idx = np.argsort(weight)
    return events[idx], weight[idx]
'''

@nb.njit
def findx(lines, events):
    rst = PointMemory(128)
    p0 = np.zeros(1, t_point)[0]
    e0 = np.zeros(1, t_event)[0]
    l0 = np.zeros(1, t_line)[0]
    # events = EventQueue(128)
    actline = ActiveLine(128)
    pts = lines.ravel()
    cursor = 0

    while events.size > 0:
        e = events.pop(events.min())
        '''
        if events.size==0 or events.min()>weight[cursor]:
            e = static[cursor]
            cursor += 1
        else: e = events.pop(events.min())
        '''
        tp, p1, p2 = e.tp, pts[e.s], pts[e.e]
        actline.sweepy = e.y
        
        # if actline.sweepy<0.3: return actline
        # print('>>> sweep y at', e.y)
        
        if tp==0: # insert
            l0['x1'], l0['y1'] = p1['x'], p1['y']
            l0['x2'], l0['y2'] = p2['x'], p2['y']
            # print('insert', l0)

            cur = actline.push(None, l0)
            
            v = actline.eval(l0)
            lidx, ridx = actline.left(v), actline.right(v)
            
            if lidx!=-1:
                l = actline.body[lidx]
                
                p = cross(l['x1'], l['y1'], l['x2'], l['y2'],
                          l0['x1'], l0['y1'], l0['x2'], l0['y2'])

                if p[1]<=e.y:
                    # print('insert > left X:', p)
                    e0['s'], e0['e'] = lidx, cur
                    e0['tp'], e0['y'] = 2, p[1]
                    v = eval_p(p[0], p[1])
                    events.push(v, e0)        
            if ridx!=-1:
                l = actline.body[ridx]
                p = cross(l0['x1'], l0['y1'], l0['x2'], l0['y2'],
                          l['x1'], l['y1'], l['x2'], l['y2'])
                if p[1]<=e.y:
                    # print('insert > right X:', p)
                    e0['s'], e0['e'] = cur, ridx
                    e0['tp'], e0['y'] = 2, p[1]
                    v = eval_p(p[0], p[1])
                    events.push(v, e0)

        if tp==1: # pop
            l0['x1'], l0['y1'] = p1['x'], p1['y']
            l0['x2'], l0['y2'] = p2['x'], p2['y']
            # print('pop', l0)
            # cur = actline.push(None, l0)
            v = actline.eval(l0)
            actline.pop(v)
            lidx, ridx = actline.left(v), actline.right(v)
            if lidx!=-1 and ridx!=-1:
                l1, l2 = actline.body[lidx], actline.body[ridx]
                p = cross(l1['x1'], l1['y1'], l1['x2'], l1['y2'],
                      l2['x1'], l2['y1'], l2['x2'], l2['y2'])
                if p[1]<=e.y:
                    # print('pop > left right X:', p)
                    e0['s'], e0['e'] = lidx, ridx
                    e0['tp'], e0['y'] = 2, p[1]
                    v = eval_p(p[0], p[1])
                    events.push(v, e0)

        if tp==2: # intersect
            actline.buf[0] = actline.body[e.s]
            actline.body[e.s] = actline.body[e.e]
            actline.body[e.e] = actline.buf[0]
            
            ll, lr = actline.body[e.s], actline.body[e.e]
            # print(ll, lr)
            v1, v2 = actline.eval(ll), actline.eval(lr)
            lidx = actline.left(min(v1, v2))
            ridx = actline.right(max(v1, v2))

            p0['x'], p0['y'] = cross(lr['x1'], lr['y1'], lr['x2'], lr['y2'],
                                     ll['x1'], ll['y1'], ll['x2'], ll['y2'])
            # print('find X', v1, v2, e.y)
            rst.push(p0)
            
            if lidx!=-1:
                l = actline.body[lidx]
                p = cross(l['x1'], l['y1'], l['x2'], l['y2'],
                          ll['x1'], ll['y1'], ll['x2'], ll['y2'])
                if p[1]<=e.y:
                    # print('intersect > left X', p)
                    e0['s'], e0['e'] = lidx, e.s
                    e0['tp'], e0['y'] = 2, p[1]
                    v = eval_p(p[0], p[1])
                    events.push(v, e0)
                    
            if ridx!=-1:
                l = actline.body[ridx]
                p = cross(lr['x1'], lr['y1'], lr['x2'], lr['y2'],
                          l['x1'], l['y1'], l['x2'], l['y2'])
                if p[1]<=e.y:
                    # print('intersect > right X', p)
                    e0['s'], e0['e'] = e.e, ridx
                    e0['tp'], e0['y'] = 2, p[1]
                    v = eval_p(p[0], p[1])
                    events.push(v, e0)
    return rst.body[:rst.size]
    

def random_lines(n, l, x_range, y_range):
    x1 = np.random.uniform(x_range[0], x_range[1], n).astype(np.float32)
    y1 = np.random.uniform(y_range[0], y_range[1], n).astype(np.float32)
    angles = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    x2 = x1 + np.cos(angles) * l
    y2 = y1 + np.sin(angles) * l
    segments = np.column_stack((x1, y1, x2, y2))
    return segments

if __name__ == '__main__':
    from time import time
    import matplotlib.pyplot as plt

    np.random.seed(3)
    lines = random_lines(300, 3, (0,10), (0,10))

    lines = lines.view(t_point).reshape(-1,2)

    events = init_events(lines)
    rst = findx(lines, events)

    
    start = time()
    events = init_events(lines)
    rst = findx(lines, events)
    print('cost', time()-start)
    
    
    
    ls = lines.view(np.float32).reshape(-1,4)
    ls = np.concatenate((ls, ls[:,:2]), axis=-1)
    ls[:,-2:] = np.nan

    plt.plot(ls[:,[0,2,4]].ravel(), ls[:,[1,3,5]].ravel())
    plt.plot(rst['x'], rst['y'], 'r.')
    plt.gca().set_aspect('equal')
    plt.show()
    

