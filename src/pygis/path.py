from pylab import *

def vlen(vec):
    """Return the Euclidean length of the vector vec."""
    return sqrt(vdot(vec, vec))

def points_on_line(start, end, step):
    """Given a start and end point, return a list containing all points on the
    line joining start to end separated by step and a list indicating the
    distance along the path this corresponds to. Note that the end point is
    included in this list.

    """

    start = array(start)
    end = array(end)
    delta = end - start
    delta_len = vlen(delta)
    alphas = hstack((arange(0, 1, step / delta_len), 1.0))
    points = [start + a * delta for a in alphas]
    dists = alphas * delta_len

    return points, dists

def path_length(points):
    """Given a list of points, return the length of the piece-wise linear path.

    """

    length = 0.0
    # for each segment...
    for start, end in zip(points[:-1], points[1:]):
        length += vlen(end-start)
    return length

# vim:filetype=python:sts=4:et:ts=4:sw=4
