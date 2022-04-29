import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

# ------------------------------------------------

class Line(art3d.Line3D):  # a line in 3D space

    def __init__(self, from_xyz=(0, 0, 0), to_xyz=(1, 1, 1), *args, **kwargs):
        xs, ys, zs = tuple(zip(from_xyz, to_xyz))
        art3d.Line3D.__init__(self, xs, ys, zs, *args, **kwargs)
        

    def location(self, from_, to_, *args):
        xs, ys, zs = tuple(zip(from_, to_))
        self.set_data_3d(xs, ys, zs)


class Point(Line):  # a point (a very short line) in 3D space

    def __init__(self, xyz=(0, 0, 0), color='black', marker='.', size=1, vanish=1.0, alpha=1.0):
        
        Line.__init__(self, xyz, xyz,
                      color=color, marker=marker, markersize=size,
                      markeredgewidth=1, linestyle='', fillstyle='none', alpha=alpha)

     
    def __fadeout(self, period, delta):
        
        def delay():
            t = time.time()
            c = 0
            while True:
                c += 1
                yield max(t + c * period - time.time(), 0)
                
        tick = delay()

        while True:
            time.sleep(next(tick))
            na = self.get_alpha() - delta
            if not self.get_visible():
                _ = self.axes
                self.remove()
                self.axes = _
                break
            if na > 0:
                self.set_alpha(na)
            else:
                self.set_visible(False)


    def location(self, at_, *args):
        Line.location(self, at_, at_)