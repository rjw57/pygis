from __future__ import print_function

import pyopencl as cl
import os
import struct
import sys
import numpy as np

context = cl.create_some_context()
command_queue = cl.CommandQueue(context)

_src = ''.join(open(os.path.join(os.path.dirname(__file__), 'kernels.cl')).readlines())
_program = cl.Program(context, _src)
_program.build()
print(_program.get_build_info(_program.devices[0], cl.program_build_info.LOG))

def hill_shade(elevation):
    """Given an elevation raster, compute a hill-shaded version with OpenCL and
    return an array with the data in it.

    """
    
    kernel = _program.image_hill_shade

    h, w = elevation.data.shape
    py, px = elevation.pixel_linear_shape

    elev_image = cl.Image(
            context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT),
            shape = (w, h),
            hostbuf = elevation.data.copy('C'))

    hs_image = cl.Image(
            context,
            cl.mem_flags.WRITE_ONLY,
            cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT),
            shape = (w, h))

    # Just to annoy us, this is the wrong blooming way round
    pixel_shape = struct.pack('ff', px, py)

    event = kernel(
            command_queue, (w, h), None,
            elev_image, hs_image, pixel_shape)

    rv = cl.enqueue_map_image(
            command_queue,
            hs_image,
            cl.map_flags.READ,
            (0,0), (w,h),
            elevation.data.shape, np.float32, 'C')

    return rv[0]
