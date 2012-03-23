import pygis.raster
import pygis.mh
import numpy as np
import scipy.stats as stats

from pathplanning import Path, Sampler

# BNG co-ords of Stonehenge and Avebury
stonehenge=(412245, 142195)
avebury=(410235, 169965)

# Load the elevation data
elev = pygis.raster.open_raster('dtm21.tif')

# Specify initial path
start_path=Path(elev.proj_to_linear((stonehenge, avebury)))

# Calculate gradient image
grad = pygis.raster.elevation_gradient(elev)

# Cost metric
def path_cost(path):
    return pygis.mh.path_cost(path.points, grad)

def plot_paths(paths):
    figure(1)
    clf()

    pygis.raster.show_raster(elev, cmap=cm.gray)
    plot(stonehenge[0], stonehenge[1], '.r', scalex=False, scaley=False)
    plot(avebury[0], avebury[1], '.r', scalex=False, scaley=False)

    for p in paths:
        a = array(p)
        plot(a[:,0], a[:,1], scalex=False, scaley=False)

def path_elevations(elev, points):
    e = None
    dists = None
    step = np.array(elev.pixel_linear_shape).min()
    for start, end in zip(points[:-1], points[1:]):
        ps, ds = pygis.path.points_on_line(start, end, step)
        es = elev.lanczos_sample(ps)
        if e is None:
            e = es
            dists = ds
        else:
            e = hstack((e, es))
            dists = hstack((dists, dists[-1] + ds))
    return dists, e

def tick(state, its=10, quiet=False):
    for i in range(its):
        state.sample()

        if not quiet and i % 10 == 0:
            print('alpha: %.3f, current: %.1f, best: %.1f' % \
                    (float(state.accepts) / state.samples, state.current[1], state.best[1]))

    if state.samples > 0:
        print('after %i iterations, alpha: %.3f, current: %.1f, best: %.1f' % \
                (state.samples, float(state.accepts) / state.samples, state.current[1], state.best[1]))

def run(state, its=100, **kwargs):
    tick(state, its, **kwargs)
    plot_paths([start_path, state.current[0].points, state.best[0].points])

#state = Sampler(start_path, grad)
#run(state, 0)

sampled_paths = []
def sample_path():
    state = Sampler(start_path, path_cost)
    for i in xrange(500):
        state.sample()

        if i % 10 == 0:
            print('alpha: %.3f, current: %.1f, best: %.1f' % \
                    (float(state.accepts) / state.samples, state.current[1], state.best[1]))
    print('after %i iterations, alpha: %.3f, current: %.1f, best: %.1f' % \
            (state.samples, float(state.accepts) / state.samples, state.current[1], state.best[1]))
    sampled_paths.append(state.current)
    # plot_paths([start_path,] + [x[0].points for x in sampled_paths])

sample_path()

# vim:filetype=python:sw=4:sts=4:et

