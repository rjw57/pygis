from osgeo import gdal
from osgeo import osr
import numpy as np
import pylab

def vlen(p):
    return np.sqrt(np.vdot(p,p))

def open_raster(filename, **kwargs):
    """Open the raster file from *filename* and return a Raster instance.
    
    Any keyword arguments are passed to the Raster() constructor.

    """

    raster = gdal.Open(filename)
    if raster is None:
        raise RuntimeError('Could not open file: %s' % (filename,))

    data = np.array(raster.ReadAsArray(), dtype=float)

    # HACK: Detect 'None' values and replace them with NAN. It's mucky :(
    data[data <= -1.0e30] = 0# np.nan

    # Form the geo-transform matrix
    gt = raster.GetGeoTransform()
    geo_transform = np.matrix([
        [gt[1], gt[2], gt[0]],
        [gt[4], gt[5], gt[3]],
        [0, 0, 1]])

    spatial_reference = osr.SpatialReference(raster.GetProjection())

    return Raster(data, spatial_reference, geo_transform, **kwargs)

def similar_raster(data, prototype, copy=False):
    """Create a raster like *prototype* but with pixel data specified by the
    array *data*. Raises a RuntimeError if the shapes of data and
    prototype.data differ in the first two elements. (The third dimension may differ.)

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
    the imshow() function except that the extent of the image is set by default
    to the linear extent of the raster rather than the pixel extent.

    For rasters with depth > 1, *plane* must specify the 0-indexed plane to display.

    **FIXME** For non axis-aligned geo-transformations, this does the wrong thing.

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
    of *raster*. This is a convenience wrapper around similar_raster(). All
    keyword arguments and the resulting raster are passed to show_raster().
    
    """

    show_raster(similar_raster(data, raster), **kwargs)

class Raster(object):
    def __init__(self, data, spatial_reference, geo_transform, copy=False, linear_scale=None):
        """Construct a new raster from pixel data and geographic information.

        *data* is an array-like object giving the per-pixel data.

        *spatial_reference* is an osr SpatialReference instance giving the
        projection used for the image plane.

        *geo_transform* is a 3x3 matrix mapping homogeneous pixel co-ordinates
        to homogeneous projection co-ordinates. **FIXME** Don't pass a
        non-affine matrix here until the code has been audited for this.

        If *copy* is True, the data will be copied, otherwise this class
        contains only a reference to the data.

        If *linear_scale* is None, the linear scale between projection
        co-ordinate and linear units (usually metres) will be read from the
        input. Not all input sources include this data. If no information is
        present, the linear scale is set to 1. Specify a linear scale directly
        here if you wish to override this behaviour.

        """

        self.data = np.array(data, copy=copy)
        self.spatial_reference = spatial_reference
        self.geo_transform = np.matrix(geo_transform, copy=True)
        self.geo_transform_inverse = np.linalg.inv(self.geo_transform)

        if linear_scale is None:
            self.linear_scale = self.spatial_reference.GetLinearUnits()
        else:
            self.linear_scale = linear_scale

        # Calculate the shape of the raster in projection co-ordinates
        bl = self.pixel_to_proj([0,0])
        br = self.pixel_to_proj([self.data.shape[1],0])
        tl = self.pixel_to_proj([0,self.data.shape[0]])
        self.proj_shape = (vlen(tl-bl), vlen(br-bl))

        # Calculate the shape of a pixel in projection co-ordinates
        self.pixel_proj_shape = (
                self.proj_shape[0] / self.data.shape[0],
                self.proj_shape[1] / self.data.shape[1]
                )

        # Calculate the shape of the raster in linear scale units
        self.linear_shape = tuple([x * self.linear_scale for x in self.proj_shape])
        self.pixel_linear_shape = tuple([x * self.linear_scale for x in self.pixel_proj_shape])

        # Calculate the extent of the raster in projection co-ords
        pix_bounds = [
                [0,0],
                [self.data.shape[1], 0],
                [0, self.data.shape[0]],
                [self.data.shape[1], self.data.shape[0]]
            ]
        proj_bounds = self.pixel_to_proj(pix_bounds)
        self.proj_extent = (
                proj_bounds[:,0].min(),
                proj_bounds[:,0].max(),
                proj_bounds[:,1].min(),
                proj_bounds[:,1].max()
            )

        self.linear_extent = tuple([x * self.linear_scale for x in self.proj_extent])

    def sample(self, points):
        """Given a list of projection space co-ordinates or a single co-ordinate,
        return the nearest-neighbour sampled pixel values. Pixel locations
        outside of the raster are clamped
        to the raster border.

        """

        # Map to pixel values
        points = self.proj_to_pixel(points)
        return self.sample_pixel(points)

    def lanczos_sample(self, points):
        pixels = self.proj_to_pixel(points)
        int_pixels = np.floor(pixels)
        a = 1
        samples = None
        norm = None
        for dx in range(-a, a+1):
            for dy in range(-a, a+1):
                coords = np.array((int_pixels[:,0] + dx, int_pixels[:,1] + dy)).transpose()
                delta = pixels - coords

                #x_kernel = np.sinc(delta[:,0]) * np.sinc(delta[:,0] / a)
                #y_kernel = np.sinc(delta[:,1]) * np.sinc(delta[:,1] / a)

                # optimised version of the kernel calculation
                k = a * np.sin(np.pi * delta) * np.sin((np.pi / a) * delta)
                denom = (np.pi*np.pi) * (delta*delta)
                k = np.where(denom != 0.0, k / denom, 1.0)

                vals = np.array(self.sample_pixel(coords))
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
        if len(points.shape) < 2 or points.shape[1] == 1:
            points = np.round(np.array([points,]))
        else:
            points = np.round(np.array(points))

        # Clamp to image
        points[:,0] = np.maximum(0, np.minimum(self.data.shape[1]-1, points[:,0]))
        points[:,1] = np.maximum(0, np.minimum(self.data.shape[0]-1, points[:,1]))

        # Sample
        values = []
        for p in points:
            values.append(self.data[p[1], p[0]])

        if len(values) == 1:
            values = values[0]
        return values

    def pixel_to_proj(self, p):
        """Given either an x,y pixel coordinate or list of co-ordinates, return
        the projection co-ordinates as a Nx2 matrix or a 1x2 vector if only one
        point was specified.

        """

        p = np.matrix(p).transpose()
        p = np.vstack((p, np.ones((1, p.shape[1]))))
        out = self.geo_transform[:2,:] * p
        out = out.transpose()
        return np.array(out)

    def proj_to_pixel(self, p):
        """Given either an x,y projection coordinate or list of co-ordinates,
        return the pixel co-ordinates as a Nx2 matrix or a 1x2 vector if only
        one point was specified.

        """

        p = np.matrix(p).transpose()
        p = np.vstack((p, np.ones((1, p.shape[1]))))
        out = self.geo_transform_inverse[:2,:] * p
        out = out.transpose()
        return np.array(out)

    def proj_to_linear(self, p):
        """Given either an x,y projection coordinate or list of co-ordinates,
        return the linear co-ordinates as a Nx2 matrix or a 1x2 vector if only
        one point was specified.

        Usually linear units are metres.

        """
        return np.array(p) * self.linear_scale

    def linear_to_proj(self, p):
        """Given either an x,y linear coordinate or list of co-ordinates,
        return the projection co-ordinates as a Nx2 matrix or a 1x2 vector if only
        one point was specified.

        """
        return np.array(p) * self.linear_scale

    def projection_wkt(self):
        """Return a string specifying the spatial reference in Well Known Text format."""
        return self.spatial_reference.ExportToPrettyWkt()

def elevation_gradient(elevation):
    """If *elevation* is a raster giving linear scale unit heights, return a
    raster with 2 planes giving, respectively, the dz/dx and dz/dy values
    measured in metre rise per horizontal metre travelled.

    """

    dx, dy = np.gradient(elevation.data)

    # Convert from metre rise / pixel run to metre rise / metre run.
    dx *= 1.0 / (elevation.pixel_linear_shape[1])
    dy *= 1.0 / (elevation.pixel_linear_shape[0])
    return similar_raster(np.dstack((dx, dy)), elevation)

def elevation_slope(elevation, grad=None):
    """If *elevation* is a raster giving linear scale unit heights, return a
    raster giving the slope as a ratio of vertical rise over horizontal run for
    each pixel.

    If *grad* is not None, it should be a gradient image as returned by
    elevation_gradient(). If None, a gradient image is calculated.

    """

    if grad is None:
        grad = elevation_gradient(elevation)

    dx = grad.data[:,:,0]
    dy = grad.data[:,:,1]
    return similar_raster(np.sqrt(dx*dx + dy*dy), elevation)

def elevation_aspect(elevation, grad=None):
    """If *elevation* is a raster giving linear scale unit heights, return a
    raster giving the aspect of the slope in radians where an aspect of zero
    points along the increasing x-direction.

    The range of the values returned is -pi to pi.

    If *grad* is not None, it should be a gradient image as returned by
    elevation_gradient(). If None, a gradient image is calculated.

    """

    if grad is None:
        grad = elevation_gradient(elevation)

    dx = grad.data[:,:,0]
    dy = grad.data[:,:,1]
    return similar_raster(np.arctan2(dy, dx), elevation)

def elevation_emboss(elevation, grad=None):
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
