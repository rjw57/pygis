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

.. note::
    Currently there is no way in which OpenCL can be enabled or disabled at
    runtime or its presence queried. The OpenCL support is still experimental
    and in flux.
