use_opencl = True

import raster

__all__ = [ 'raster' ]

import os
def data_file(filename):
    f = os.path.join(os.path.dirname(__file__), 'data', filename)
    if os.path.exists(f):
        return f
    raise RuntimeError('No such data file: %s' % (filename,))
