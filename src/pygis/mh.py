from pylab import *
import pygis.path

def vlen(vec):
    """Return the Euclidean length of the vector vec."""
    return sqrt(vdot(vec, vec))

def line_cost(start, end, grad):
    """Given some gradient raster, calculate the cost of the line start -> end.
    
    """

    start = array(start)
    end = array(end)
    delta = end - start
    delta_len = vlen(delta)
    direction = delta / delta_len

    step = array(grad.pixel_linear_shape).min()
    line_points, dists = pygis.path.points_on_line(start, end, step)
    return integrate_cost(line_points, grad)

def integrate_cost(points, grad):
    """Given some gradient raster, calculate the cost of the lines segments
    joining the points assuming that the size of each line segment is small
    w.r.t pixel size.
    
    """

    points = np.array(points)
    assert len(points.shape) > 1 and points.shape[0] > 1

    midpoints = 0.5*(points[:-1] + points[1:])
    deltas = points[1:] - points[:-1]
    delta_lengths = np.sqrt((deltas * deltas).sum(axis=1)).transpose()
    directions = np.array(deltas, copy=True)
    directions[:,0] /= delta_lengths
    directions[:,1] /= delta_lengths

    samples = grad.lanczos_sample(midpoints)
    if len(points) == 2:
        samples = [samples,]

    cost = 0.0
    for direction, dh, gradient in zip(directions, delta_lengths, samples):
        # get vert. distance in metres
        dv = vdot(gradient, direction) * dh

        # get linear distance
        d = vlen(array(dh, dv))

        # calculate the subjective gradient
        subj_gradient = vdot(gradient, direction)

        # increase cost
        cost += abs(subj_gradient) * d

    return cost

def path_cost(points, grad):
    """Given a list of points specifying a path and a gradient raster, return the total cost of the path.

    """

    step = array(grad.pixel_linear_shape).min()
    int_points = []
    for start, end in zip(points[:-1], points[1:]):
        np, _ = pygis.path.points_on_line(start, end, step)
        int_points.extend(np[:-1])
    int_points.append(points[-1])
    return integrate_cost(int_points, grad)

# vim:filetype=python:sts=4:et:ts=4:sw=4
