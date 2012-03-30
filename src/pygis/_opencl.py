from __future__ import print_function

import pygis

_have_opencl = True

try:
    import pyopencl as cl
except ImportError:
    _have_opencl = False

import os
import struct
import sys
import numpy as np

class OpenCLNotPresentError(Exception):
    pass

_context = None
def context():
    if not pygis.use_opencl:
        raise OpenCLNotPresentError()

    global _context
    if _context is not None:
        return _context
    elif _have_opencl:
        _context = cl.create_some_context()
        return _context

    raise OpenCLNotPresentError()

_command_queue = None
def command_queue():
    if not pygis.use_opencl:
        raise OpenCLNotPresentError()

    global _command_queue
    if _command_queue is not None:
        return _command_queue
    elif _have_opencl:
        _command_queue = cl.CommandQueue(context())
        return _command_queue

    raise OpenCLNotPresentError()

_program = None
def program():
    if not pygis.use_opencl:
        raise OpenCLNotPresentError()

    global _program
    if _program is not None:
        return _program
    elif _have_opencl:
        src = ''.join(open(pygis.data_file('kernels.cl')).readlines())
        _program = cl.Program(context(), src)
        _program.build()
        print(_program.get_build_info(_program.devices[0], cl.program_build_info.LOG))
        return _program

    raise OpenCLNotPresentError()

def hill_shade(elevation):
    """Given an elevation raster, compute a hill-shaded version with OpenCL and
    return an array with the data in it.

    """

    kernel = program().image_hill_shade
    ctx = context()
    cq = command_queue()

    h, w = elevation.data.shape
    px, py = elevation.pixel_linear_shape

    elev_image = cl.Image(
            ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT),
            shape = (w, h),
            hostbuf = elevation.data.copy('C'))

    hs_image = cl.Image(
            ctx,
            cl.mem_flags.WRITE_ONLY,
            cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT),
            shape = (w, h))

    pixel_shape = struct.pack('ff', px, py)
    event = kernel(cq, (w, h), None, elev_image, hs_image, pixel_shape)

    rv = cl.enqueue_map_image(
            cq,
            hs_image,
            cl.map_flags.READ,
            (0,0), (w,h),
            elevation.data.shape, np.float32, 'C',
            wait_for = (event,),
            is_blocking = True)

    data = rv[0].copy()
    rv[0].base.release(cq)

    return data
