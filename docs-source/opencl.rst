OpenCL acceleration
===================

If installed, the :py:mod:`pyopencl` module will be used to accelerate some
operations with OpenCL. If OpenCL is not available, or if the
:py:mod:`pyopencl` module is not installed, a software fallback implemented via
numpy will be used instead.

Generally the OpenCL acceleration should be faster even if your OpenCL
implementation is software only since OpenCL allows an algorithm to be compiled
in its entirety to machine code whereas the numpy implementation will always
have to restrict itself to the numpy operations.

Disabling OpenCL
----------------

If the :py:mod:`pyopencl` module could be loaded, OpenCL will automatically be
used. Setting :py:attr:`pygis.use_opencl` to *False* can be used to override this
behaviour. The default behaviour can be restored by setting this attribute back
to *True*.
