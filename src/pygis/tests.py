import pygis
import unittest
import numpy as np

class TestRaster(unittest.TestCase):
    def setUp(self):
        self.mons = pygis.raster.open_raster(pygis.data_file('olympus-mons.tiff'))

    def test_raster_size(self):
        self.assertEquals(self.mons.data.shape, (523, 542))

    def test_projective_size(self):
        self.assertEquals(self.mons.proj_extent, (2231012, 3235338, 618902, 1588021))

    def test_pixel_size(self):
        self.assertEquals(self.mons.pixel_proj_shape, (1853, 1853))

    def test_opencl_hill_shade(self):
        pygis.use_opencl = False
        a = pygis.raster.elevation_hill_shade(self.mons).data
        pygis.use_opencl = True
        b = pygis.raster.elevation_hill_shade(self.mons).data
        self.assertTrue(np.abs(a-b).max() < 1e-5)

def test_suite():
    return unittest.makeSuite(TestRaster)
