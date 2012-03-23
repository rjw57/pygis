from pylab import *
import numpy as np
import pygis.path

def integrate_cost(points, grad):
    """Given some gradient raster, calculate the cost of the lines segments
    joining the points assuming that the size of each line segment is small
    w.r.t pixel size.
    
    """

    points = np.array(points)
    assert len(points.shape) > 1 and points.shape[0] > 1

    midpoints = 0.5*(points[:-1] + points[1:])
    directions = points[1:] - points[:-1]
    segment_lengths = np.sqrt((directions * directions).sum(axis=1)).transpose()
    directions[:,0] /= segment_lengths
    directions[:,1] /= segment_lengths

    samples = grad.lanczos_sample(midpoints)
    if len(points) == 2:
        samples = [samples,]
    samples = np.array(samples)
    
    # samples: Nx2 array of gradient samples
    # directions: Nx2 array of normalised segment directions
    # segment_lengths: Nx2 array of segment horizontal lengths

    # calculate subjective slope
    subj_slope = (samples * directions).sum(axis=1)

    # hence vertical displacement
    vdisp = subj_slope * segment_lengths

    # hence Euclidean displacement
    disp = np.sqrt(vdisp*vdisp + segment_lengths*segment_lengths)

    # cost is distance * absolute slope
    cost = (disp * np.abs(subj_slope)).sum()

    return cost

def path_cost(points, grad):
    """Given a list of points specifying a path and a gradient raster, return the total cost of the path.

    """

    step = np.abs(array(grad.pixel_proj_shape)).min()
    int_points = []
    for start, end in zip(points[:-1], points[1:]):
        pts, _ = pygis.path.points_on_line(start, end, step)
        int_points.extend(pts[:-1])
    int_points.append(points[-1])
    return integrate_cost(int_points, grad)

def line_cost(start, end, grad):
    """Given some gradient raster, calculate the cost of the line start -> end.
    
    """

    start = array(start)
    end = array(end)
    delta = end - start
    delta_len = np.sqrt(np.dot(delta, delta))
    direction = delta / delta_len

    step = np.abs(array(grad.pixel_proj_shape)).min()
    line_points, dists = pygis.path.points_on_line(start, end, step)
    return integrate_cost(line_points, grad)

# vim:filetype=python:sts=4:et:ts=4:sw=4
