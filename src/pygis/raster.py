"""
Handling raster data
====================

A raster is a two-dimensional array of pixels where each pixel is associated
with a particular location in some projection co-ordinate system. Each pixel
may contain one or more 'bands' of data.

For example, a digital terrain model will usually consist only of one band of
data corresponding to the elevation at a particular projection co-ordinate.

The raster data itself is represented by a :py:mod:`numpy`-compatible array.
The :py:class:`Raster` class defines a wrapper around an array which maintains
other geographic metadata such as the extend of the image in projection
co-ordinates and the conversion (if any) between projection co-ordinates and
linear scales.

"""

from osgeo import gdal
from osgeo import osr
import numpy as np
import pylab

from _opencl import hill_shade as cl_hill_shade
from _opencl import OpenCLNotPresentError

class Raster(object):
    r"""A wrapper around a two-dimensional raster dataset.

    For data with `B` bands, `N` vertical pixels and `M` horizontal pixels, the
    wrapped data is an array whose shape is `N \times M \times B`.

    :param data: an array-like object giving the per-pixel data.
    :param spatial_reference: the projection used for the image plane.
    :type spatial_reference: :py:class:`osr.SpatialReference`
    :param geo_transform: a matrix mapping homogeneous pixel co-ordinates to homogeneous projection co-ordinates.
    :param copy: if True, the data will be copied, otherwise this class contains only a reference to the data.
    :param linear_scale: the muliplicative scaling between projection co-ordinates and linear units.
    
    .. warning::
        Don't pass a non-affine matrix to *geo_transform* until the code has
        been audited for this.

    If *linear_scale* is None, the linear scale between projection co-ordinate
    and linear units (usually metres) will be read from *spatial_reference*.
    Not all references include this data. If no information is present, the
    linear scale is set to 1. Specify a linear scale directly here if you wish
    to override this behaviour.

    """

    def __init__(self, data, spatial_reference, geo_transform, copy=False, linear_scale=None):
        #: The array-like object wrapped by this :py:class:`Raster`.
        self.data = np.array(data, copy=copy)

        #: The spatial reference giving the projection used.
        self.spatial_reference = spatial_reference

        #: A 3 `\times` 3 matrix mapping homogeneous pixel co-ordinates to projection co-ordinates. 
        self.geo_transform = np.matrix(geo_transform, copy=True)

        #: The inverse of :py:attr:`geo_transform`.
        self.geo_transform_inverse = np.linalg.inv(self.geo_transform)

        #: The multiplicative scaling between projection co-ordinates and linear units (usually metres).
        self.linear_scale = self.spatial_reference.GetLinearUnits() if linear_scale is None else linear_scale

        # Calculate the shape of the raster in projection co-ordinates
        bl = self.pixel_to_proj([0,0])
        br = self.pixel_to_proj([self.data.shape[1],0])
        tl = self.pixel_to_proj([0,self.data.shape[0]])

        def vlen(p):
            return np.sqrt(np.vdot(p,p))

        #: A tuple giving the width and height of the raster in projection co-ordinates.
        self.proj_shape = (vlen(br-bl), vlen(tl-bl))

        #: A tuple giving the (absolute) width and height of one pixel in projection co-ordinates.
        self.pixel_proj_shape = (
            np.abs(self.proj_shape[0] / self.data.shape[1]),
            np.abs(self.proj_shape[1] / self.data.shape[0])
        )

        # Calculate the shape of the raster in linear scale units

        #: A tuple giving the width and height of the raster in linear units.
        self.linear_shape = tuple([x * self.linear_scale for x in self.proj_shape])

        #: A tuple giving the (absolute) width and height of one pixel in linear units.
        self.pixel_linear_shape = tuple([x * self.linear_scale for x in self.pixel_proj_shape])

        # Calculate the extent of the raster in projection co-ords
        pix_bounds = [
                [0,0],
                [self.data.shape[1], self.data.shape[0]]
            ]
        proj_bounds = self.pixel_to_proj(pix_bounds)

        #: A tuple giving the left, right, bottom and top of the image in projection co-ordinates.
        self.proj_extent = (
                proj_bounds[0,0],
                proj_bounds[1,0],
                proj_bounds[1,1],
                proj_bounds[0,1],
            )

        #: A tuple giving the left, right, bottom and top of the image in linear units.
        self.linear_extent = tuple([x * self.linear_scale for x in self.proj_extent])

    def sample(self, points):
        r"""Given a `N \times 2` array of projection space co-ordinates or a
        single return a `N \times 1` array of the nearest-neighbour sampled
        pixel values. Pixel locations outside of the raster are clamped to the
        raster border. Integer pixel locations are considered to be at the
        centre of a pixel.

        The projection co-ordinates are specified in the order *column*, *row*.

        """

        # Map to pixel values
        points = self.proj_to_pixel(points)
        return self.sample_pixel(points)

    def lanczos_sample(self, points):
        r"""Given a `N \times 2` array of projection space co-ordinates or a
        single return a `N \times 1` array of the Lanczos sampled pixel values.
        Pixel locations outside of the raster are clamped to the raster border.
        Integer pixel locations are considered to be at the centre of a pixel.

        Lanczos sampling is slower than nearest-neighbour but will provide
        'sensible' smoothed values for non-integer pixel co-ordinates.

        The projection co-ordinates are specified in the order *column*, *row*.

        """

        pixels = self.proj_to_pixel(points)
        int_pixels = np.floor(pixels)
        a = 1
        samples = None
        norm = None
        for dx in range(-a, a+1):
            for dy in range(-a, a+1):
                coords = np.array((int_pixels[:,0] + dx, int_pixels[:,1] + dy)).transpose()
                delta = pixels - coords

                # optimised version of the kernel calculation
                k = a * np.sin(np.pi * delta) * np.sin((np.pi / a) * delta)
                denom = (np.pi*np.pi) * (delta*delta)
                k = np.where(denom != 0.0, k / denom, 1.0)

                vals = self.sample_pixel(coords)
                kernel = k.prod(axis=1)#x_kernel * y_kernel

                if len(vals.shape) > 1 and vals.shape[1] > 1:
                    kernel = np.tile(kernel, (vals.shape[1],1)).transpose()

                contribs = kernel * vals
                
                if samples is None:
                    samples = contribs
                    norm = kernel
                else:
                    samples += contribs
                    norm += kernel

        return samples / norm

    def sample_pixel(self, points):
        # Round the points to the integer pixel locations.
        points = np.round(points)

        # Clamp to image
        points[:,0] = np.maximum(0, np.minimum(self.data.shape[1]-1, points[:,0]))
        points[:,1] = np.maximum(0, np.minimum(self.data.shape[0]-1, points[:,1]))

        # Sample
        values = []
        for p in points:
            values.append(self.data[p[1], p[0]])

        return np.array(values)

    def pixel_to_proj(self, p):
        r"""Given a `N \times 2` array of pixel co-ordinates, return the
        corresponding projection co-ordinates as a `N \times 2` array.

        The pixel co-ordinates are specified in the order *column*, *row*.

        """

        p = np.matrix(p).transpose()
        p = np.vstack((p, np.ones((1, p.shape[1]))))
        out = self.geo_transform[:2,:] * p
        out = out.transpose()
        return np.array(out)

    def proj_to_pixel(self, p):
        r"""Given a `N \times 2` array of projection co-ordinates, return the
        corresponding pixel co-ordinates as a `N \times 2` array.

        The pixel co-ordinates are specified in the order *row*, *column*.

        """

        p = np.matrix(p).transpose()
        p = np.vstack((p, np.ones((1, p.shape[1]))))
        out = self.geo_transform_inverse[:2,:] * p
        out = out.transpose()
        return np.array(out)

    def projection_wkt(self):
        """Return a string specifying the spatial reference in Well Known Text format."""
        return self.spatial_reference.ExportToPrettyWkt()

def open_raster(filename, window=None, **kwargs):
    """Open a raster file from *filename* and return a Raster instance.

    :param window: the window, in projection co-ordinates, to load.

    If *window* is not None, it is a tuple specifying the minimum x, maximum x,
    minimum y, maximum y window to load from the raster. The co-ordinates are
    specified as projection co-ordinates. If the requested region is outside of
    the dataset bounds, a RuntimeError is raised. The requested region is
    expanded to cover an integer number of pixels.
    
    Any keyword arguments are passed to the :py:class:`Raster` constructor.

    This function uses the :py:mod:`gdal` module in its implementation. Any
    raster format which GDAL can open can be opened by this function.

    """

    raster = gdal.Open(filename)
    if raster is None:
        raise RuntimeError('Could not open file: %s' % (filename,))

    xoff, yoff = (0, 0)
    xsize, ysize = (None, None)

    # Form the geo-transform matrix
    gt = raster.GetGeoTransform()
    geo_transform = np.matrix([
        [gt[1], gt[2], gt[0]],
        [gt[4], gt[5], gt[3]],
        [0, 0, 1]])

    if window is not None:
        proj_extent = np.array([
            [window[0], window[2], 1],
            [window[1], window[3], 1]
        ]).transpose()
        pix_extent = (np.linalg.inv(geo_transform) * proj_extent).transpose()

        if (pix_extent < 0).any() or \
           (pix_extent[:,0] >= raster.RasterXSize).any() or \
           (pix_extent[:,1] >= raster.RasterYSize).any():
               raise RuntimeError('Window out of bounds')

        offset = np.array(pix_extent.min(axis=0))[0]
        size = np.abs(np.array(pix_extent[1] - pix_extent[0])[0])
        xoff, yoff, _ = [int(x) for x in offset]
        xsize, ysize, _ = [int(x) for x in size]

    proj_off = geo_transform * np.matrix([xoff, yoff, 1]).transpose()
    geo_transform[0:2,2] = proj_off[0:2]

    data = np.array(raster.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize), dtype=np.float32)
    if len(data.shape) > 2:
        # cope with odd ordering of arrays
        # 0,1,2 -> 1,2,0
        data = np.swapaxes(data, 0, 1)
        data = np.swapaxes(data, 1, 2)

    # HACK: Detect 'None' values and replace them with NAN. It's mucky :(
    data[data <= -1.0e30] = 0# np.nan

    spatial_reference = osr.SpatialReference(raster.GetProjection())

    return Raster(data, spatial_reference, geo_transform, **kwargs)

def write_raster(raster, filename):
    """Write a raster to filename in the GeoTIFF format."""

    drv = gdal.GetDriverByName('GTiff')
    n_bands = raster.data.shape[2] if len(raster.data.shape) > 2 else 1
    ds = drv.Create(filename, raster.data.shape[1], raster.data.shape[0], n_bands, gdal.GDT_Float32)
    ds.SetProjection(raster.projection_wkt())

    gt = raster.geo_transform
    gt_list = [gt[0,2], gt[0,0], gt[0,1], gt[1,2], gt[1,0], gt[1,1]]
    ds.SetGeoTransform(gt_list)

    for band_idx in xrange(n_bands):
        band = ds.GetRasterBand(band_idx+1)
        band.SetNoDataValue(np.nan)
        if len(raster.data.shape) > 2:
            band.WriteArray(raster.data[:,:,band_idx])
        else:
            band.WriteArray(raster.data)

def similar_raster(data, prototype, copy=False):
    """Create a raster like *prototype* but with pixel data specified by the
    array *data*. Raises a RuntimeError if the shapes of data and
    *prototype.data* differ in the first two elements. (The third dimension may
    differ.)

    If *copy* is True, a copy of the data is made.

    """
    if data.shape[:2] != prototype.data.shape[:2]:
        raise RuntimeError('Prototype and data must have equivalent shape. ' + 
                           'Got %s and %s respectively.' % (data.shape, prototype.data.shape))

    return Raster(
            data, prototype.spatial_reference, prototype.geo_transform,
            copy=copy, linear_scale=prototype.linear_scale)

def show_raster(raster, plane=None, **kwargs):
    """Show the raster image *raster* on the current axes. This behaves like
    the pylab imshow() function except that the extent of the image is set by default
    to the projection co-ordinate extent of the raster rather than the pixel extent.

    For rasters with depth > 1, *plane* must specify the 0-indexed plane to display.

    .. warning::
        For non axis-aligned geo-transformations, this does the wrong thing.

    """

    if 'extent' not in kwargs:
        kwargs['extent'] = raster.proj_extent

    if len(raster.data.shape) < 3 or raster.data.shape[2] == 1:
        pylab.imshow(raster.data, **kwargs)
    elif plane is not None:
        pylab.imshow(raster.data[:,:,plane], **kwargs)
    else:
        raise RuntimeError('You must specify which plane to display for multi-plane rasters.')

def show_raster_data(data, raster, **kwargs):
    """Show the raster data *data* on the current axes as if it were the data
    of *raster*. This is a convenience wrapper around
    :py:func:`similar_raster`. All keyword arguments and the resulting raster
    are passed to :py:func:`show_raster`.
    
    """

    show_raster(similar_raster(data, raster), **kwargs)

def elevation_gradient(elevation):
    """Calculate the two-dimensional gradient vector for an elevation raster.
    
    :param elevation: a raster giving linear scale unit heights.
    
    Return a raster with 2 planes giving, respectively, the dz/dx and dz/dy
    values measured in metre rise per horizontal metre travelled.

    """

    dx, dy = np.gradient(elevation.data)

    # Convert from metre rise / pixel run to metre rise / metre run.
    dx *= 1.0 / (elevation.pixel_linear_shape[1])
    dy *= 1.0 / (elevation.pixel_linear_shape[0])
    return similar_raster(np.dstack((dx, dy)), elevation)

def elevation_slope(elevation, grad=None):
    """Calculate the rise-over-run slope from an elevation raster.
    
    :param elevation: a raster giving linear scale unit heights.
    
    Return a raster giving the slope as a ratio of vertical rise over
    horizontal run for each pixel.

    If *grad* is not None, it should be a gradient image as returned by
    :py:func:`elevation_gradient`. If None, a gradient image is calculated.

    """

    if grad is None:
        grad = elevation_gradient(elevation)

    dx = grad.data[:,:,0]
    dy = grad.data[:,:,1]
    return similar_raster(np.sqrt(dx*dx + dy*dy), elevation)

def elevation_aspect(elevation, grad=None):
    r"""Calculate the aspect (direction of greatest slope) from an elevation raster.
    
    :param elevation: a raster giving linear scale unit heights.

    Return a raster giving the aspect of the slope in radians where an aspect
    of zero points along the increasing x-direction.  The range of the values
    returned is `-\pi` to `\pi`.

    If *grad* is not None, it should be a gradient image as returned by
    :py:func:`elevation_gradient`. If None, a gradient image is calculated.

    """

    if grad is None:
        grad = elevation_gradient(elevation)

    dx = grad.data[:,:,0]
    dy = grad.data[:,:,1]
    return similar_raster(np.arctan2(dy, dx), elevation)

def elevation_hill_shade(elevation, grad=None):
    r"""Render a hill-shaded image of an elevation raster.
    
    :param elevation: a raster giving linear scale unit heights.

    If *grad* is not None, it should be a gradient image as returned by
    :py:func:`elevation_gradient`. If None, a gradient image is calculated.

    If OpenCL acceleration is in use, the *grad* parameter is ignored.

    """

    try:
        return similar_raster(
            cl_hill_shade(elevation),
            elevation,
            copy=True)
    except OpenCLNotPresentError:
        if grad is None:
            grad = elevation_gradient(elevation)

        shape = grad.data.shape[:2]
        dxvs = np.dstack((np.ones(shape), np.zeros(shape), grad.data[:,:,0]))
        dyvs = np.dstack((np.zeros(shape), np.ones(shape), grad.data[:,:,1]))

        mdx=np.sqrt(np.sum(dxvs*dxvs, axis=2))
        mdy=np.sqrt(np.sum(dyvs*dyvs, axis=2))

        for i in range(3):
            dxvs[:,:,i] /= mdx
            dyvs[:,:,i] /= mdy

        norms = np.cross(dxvs, dyvs)
        norms_len = np.sqrt(np.sum(norms*norms, axis=2))
        for i in range(3):
            norms[:,:,i] /= norms_len

        light = np.array([-1,-1,0.3])
        light /= np.sqrt(np.dot(light,light))

        for i in range(3):
            norms[:,:,i] *= light[i]

        return similar_raster(np.maximum(0, np.sum(norms, axis=2)), grad)

# vim:sw=4:sts=4:et
