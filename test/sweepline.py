import sys; sys.path.append('../')

import numpy as np
import numba as nb

from structron import TypedAVLTree, TypedRBTree, TypedHeap
from structron import TypedMemory



t_point = np.dtype([('x', np.int64), ('y', np.int64)])
t_line = np.dtype([('x1', np.int64), ('y1', np.int64),
                   ('x2', np.int64), ('y2', np.int64)])

t_event = np.dtype([('x', np.float64), ('y', np.float64),
                    ('tp', np.int8), ('s', np.int32), ('e', np.int32)])

@nb.njit
def line_cross(actline, i1, i2, e0):
    if i1 == -1 or i2 == -1: return -1
    l1, l2 = actline.body[i1], actline.body[i2]
    x1, y1, x2, y2 = l1['x1'], l1['y1'], l1['x2'], l1['y2']
    x3, y3, x4, y4 = l2['x1'], l2['y1'], l2['x2'], l2['y2']

    dx12, dy12, dx34, dy34 = x1 - x2, y1 - y2, x3 - x4, y3 - y4
    denom = dx12 * dy34 - dy12 * dx34

    if denom == 0: return -1
    t = (x1 - x3) * dy34 - (y1 - y3) * dx34
    u = dx12 * (y1 - y3) - dy12 * (x1 - x3)

    if not (0 <= t <= denom or 0 >= t >= denom): return -1
    if not (0 <= -u <= denom or 0 >= -u >= denom): return -1
    x = (x1 * denom - t * dx12) / denom
    y = (y1 * denom - t * dy12) / denom

    e0['s'], e0['e'] = i1, i2
    e0['tp'], e0['x'], e0['y'] = 2, x, y
    
    if y > actline.sweepy: return -1
    if y < actline.sweepy: return 1
    if y == actline.sweepy:
        if x > actline.sweepx: return 1
        if x == actline.sweepx: return 0
    return -1

def comp_e(self, e1, e2):
    if e1.y != e2.y: return e2.y - e1.y
    if e1.x != e2.x: return e1.x - e2.x
    if e1.tp != e2.tp: return e1.tp - e2.tp
    if e1.s != e2.s: return e1.s - e2.s
    if e1.e != e2.e: return e1.e - e2.e
    return 0

def comp_l(self, l1, l2):
    def eval_x(self, l):
        if l['y1']==l['y2']: return self.sweepx
        dx = l['x1'] * (l['y2'] - l['y1'])
        dx += (l['x2'] - l['x1']) * (self.sweepy - l['y1'])
        return dx / (l['y2'] - l['y1'])

    def angle_x(self, l):
        dx = l['x1'] - l['x2']
        dy = l['y1'] - l['y2']
        norm = (dx**2 + dy**2)**0.5
        return dx / norm * self.inout

    d = eval_x(self, l1) - eval_x(self, l2)
    if d!=0: return d
    return angle_x(self, l1) - angle_x(self, l2)

def print_tree(tree, mar=3, bal=False):
    nodes = [tree.root]
    rst = []
    while max(nodes)!=-1:
        cur = nodes.pop(0)
        if cur==-1:
            rst.append(' ')
            nodes.extend([-1, -1])
            continue
        ilrk = tree.idx[cur]
        value = ilrk['id'] - 1
        if bal: value = (value, ilrk['bal'])
        rst.append(value)
        nodes.append(ilrk['left'])
        nodes.append(ilrk['right'])

    def fmt(s, width):
        s = '%s'%str(s)
        l = len(s)
        pad = ' '*((width+1-l)//2)
        return (pad + s + pad)[:width]
    
    levels = int(np.ceil(np.log(len(rst)+1)/np.log(2)))       
    s = 0
    for r in range(levels):
        width = mar * 2 ** (levels - r - 1)
        line = [fmt(i, width) for i in rst[s:s+2**r]]
        print(''.join(line))
        s += 2 ** r
    print()

ActiveLine = TypedAVLTree(comp_l, t_line,
    {'sweepy':np.float64, 'sweepx':np.float64, 'inout':np.int32})
EventQueue = TypedAVLTree(comp_e, t_event)
EventHeap = TypedHeap(comp_e, t_event)
PointMemory = TypedMemory(t_point)

@nb.njit
def init_events(lines):
    e0 = np.zeros(1, t_event)[0]
    pts = lines.ravel()
    events = EventQueue(128)
    # events = np.zeros(len(pts), dtype=t_event)

    # 1: insert, 2: cross, 3: pop
    for i in range(0, len(pts), 2):
        p1, p2 = pts[i], pts[i+1]

        if p1['y'] != p2['y']:
            dir = p2['y'] - p1['y']
        else: dir = p1['x'] - p2['x']

        dir = dir < 0
        e0['s'] = i if dir else i+1
        e0['e'] = i+1 if dir else i
        
        e0['tp'] = 1
        e0['x'] = p1['x'] if dir else p2['x']
        e0['y'] = p1['y'] if dir else p2['y']
        events.push(e0)
        
        e0['tp'] = 3
        e0['x'] = p2['x'] if dir else p1['x']
        e0['y'] = p2['y'] if dir else p1['y']
        events.push(e0)
    return events

debug = False
@nb.njit
def findx(lines, events):
    rst = PointMemory(128)
    e0 = np.zeros(1, t_event)[0]
    p0 = np.zeros(1, t_point)[0]
    l0 = np.zeros(1, t_line)[0]
    xs = np.zeros(128, dtype=np.int32); xn = 0
    series = xs[64:]; sn = 0
    cursor = 0
    
    actline = ActiveLine(128)
    # events = EventQueue(128)
    
    pts = lines.ravel()
    '''
    while events.size > 0 or termins.size>0:
        midx = events.min()
        e = events.body[midx]
        if events.size==0 or events.comp(termins.top(), e)<0:
            # e = termins[cursor]; cursor += 1
            e = termins.pop()
        else: events.pop(e)
    '''
    
    while events.size > 0:
        midx = events.min()
        e = events.body[midx]
        events.pop(e)
        
        actline.sweepx, actline.sweepy = e['x'], e['y']

        if debug:
            print('\n\n>>>>>>>>>> sweep x,y at:', actline.sweepx, actline.sweepy)
            print('\t event:', e)
            
        if e['tp']==1: # insert
            actline.inout = 1
            p1, p2 = pts[e['s']], pts[e['e']]
            l0['x1'], l0['y1'] = p1['x'], p1['y']
            l0['x2'], l0['y2'] = p2['x'], p2['y']
            # print('insert', l0)
            
            cur = actline.push(l0)
            if debug: print('insert', l0, 'at', cur)
            
            lidx, ridx = actline.left(l0), actline.right(l0)
            # remove left right X
            if line_cross(actline, lidx, ridx, e0)>=0:
                if debug: print('pop > left right X:', e0)
                events.pop(e0)
            # insert left X (if exist)
            if line_cross(actline, lidx, cur, e0)>=0: 
                if debug: print('insert > left X:', e0)
                events.push(e0)
            # insert right X (if exist)
            if line_cross(actline, cur, ridx, e0)>=0:
                if debug: print('insert > right X:', e0)
                events.push(e0)
        
        if e['tp']==3: # pop
            actline.inout = -1
            p1, p2 = pts[e['s']], pts[e['e']]
            l0['x1'], l0['y1'] = p1['x'], p1['y']
            l0['x2'], l0['y2'] = p2['x'], p2['y']
            # pop event, and check left-right X
            
            if debug: print('pop line', actline.index(l0), l0)
            actline.pop(l0)
            
            lidx, ridx = actline.left(l0), actline.right(l0)
            if line_cross(actline, lidx, ridx, e0)>0:
                if debug: print('push > left X:', e0)
                events.push(e0)
        
        if e['tp']==2: # intersect
            
            p0['x'], p0['y'] = e['x'], e['y']
            rst.push(p0)
            if debug: print('find X:', p0)
            
            actline.inout = 1
            le = e
            sn = xn = 0

            # add all X events at same point
            while True:
                xs[xn] = le['s']; xn+=1
                xs[xn] = le['e']; xn+=1
                midx = events.min()
                nexte =  events.body[midx]
                if events.size==0: break
                if le['x']!=nexte['x']: break
                if le['y']!=nexte['y']: break
                if nexte['tp']!=2: break
                events.pop(nexte)
                le = nexte
            
            # collect xs lines
            l0['x1'], l0['x2'] = le['x']-1, le['x']-1
            l0['y1'], l0['y2'] = le['y']-1, le['y']

            firstx = lastx = -1
            minid = actline.left(l0)
            series[sn] = minid; sn+=1

            actline.min() if minid==-1 else actline.right(l0)

            
            while True:
                cur = actline.next()
                if cur in xs[:xn]:
                    series[sn] = cur; sn+=1
                    break
                series[0] = cur

            while True:
                cur = actline.next()
                series[sn] = cur; sn+=1
                if not cur in xs[:xn]: break

            if debug: print(xs[:xn], series[:sn])

            lidx, ridx = series[0], series[sn-1]

            # pop most left right X
            if line_cross(actline, lidx, series[1], e0)>0:
                if debug: print('pop > left X:', e0)
                events.pop(e0)
            if line_cross(actline, series[sn-2], ridx, e0)>0:
                if debug: print('pop > right X:', e0)
                events.pop(e0)

            # swap X lines sequence
            for si in range((sn-2)//2):
                idx1, idx2 = series[si+1], series[sn-si-2]
                l1, l2 = actline.body[idx1], actline.body[idx2]
                l1['x1'], l2['x1'] = l2['x1'], l1['x1']
                l1['y1'], l2['y1'] = l2['y1'], l1['y1']
                l1['x2'], l2['x2'] = l2['x2'], l1['x2']
                l1['y2'], l2['y2'] = l2['y2'], l1['y2']

            # insert most left right X
            if debug: print('left right')
            if line_cross(actline, lidx, series[1], e0)>0:
                if debug: print('insert > left X:', e0)
                events.push(e0)
            if line_cross(actline, series[sn-2], ridx, e0)>0:
                if debug: print('insert > right X:', e0)
                events.push(e0)
        if debug: print_tree(actline)
        # input('round ...')
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

    '''
    np.random.seed(0)

    l0 = np.array([[-1,2],[-1,-2],[0,0],[0,1], [-1,2],[1,-2], [-1,-1],[1,1], [0,0],[-1,1], [0,0],[-1,-2]], dtype=np.float32)

    l0 = np.array([(-3,0,3,0),(-3,3,3,-3),(0,-3,0,3)], dtype=np.float32).reshape(-1,4)
    l0 = np.concatenate([l0, l0+1, l0+2, l0+3])
    l0 = l0 * 10000 + np.random.randint(-2, 2, l0.shape)
    '''
    
    np.random.seed(3)
    l0 = random_lines(1000000, 0.1, (0,100), (0,100))

    # l0 = np.array([[-1,2],[-1,-2],[0,0],[0,1], [-1,2],[1,-2], [-1,-1],[1,1], [0,0],[-1,1], [0,0],[-1,-2]], dtype=np.float32)
    l0 *= 1e5

    from shapely.geometry import MultiLineString
    from shapely.ops import unary_union
    
    ls = MultiLineString(list(l0.reshape(-1,2,2)))
    
    start = time()
    # unary_union(ls.geoms)
    rst = print(time()-start)
        
    lines = l0.astype(np.int64).view(t_point).reshape(-1,2)
    
    events = init_events(lines)
    rst = findx(lines, events)

    start = time()
    events = init_events(lines)
    rst = findx(lines, events)
    print(len(rst), 'pints, cost', time()-start)
    print(len(rst), 'x found')

    ls = lines.view(np.int64).astype(np.float64).reshape(-1,4)
    ls = np.concatenate((ls, ls[:,:2]), axis=-1)
    ls[:,-2:] = np.nan
    
    plt.plot(ls[:,[0,2,4]].ravel(), ls[:,[1,3,5]].ravel())
    plt.plot(rst['x'], rst['y'], 'r.')
    plt.gca().set_aspect('equal')
    plt.show()

    
        

    



