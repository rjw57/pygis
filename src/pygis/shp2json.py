from __future__ import print_function

import anyjson
import argparse
import logging
import os
import osgeo.osr as osr
import shapefile

log = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def describe_shapefile(shp, dbf):
    """Form a dictionary describing the joint contents of a shp file and
    (optionally) a dbf file.

    *shp* and *dbf* should be file-like objects for reading the appropriate
    data or *None* if the data is not available. If *shp* is None, a
    *ValueError* is raised.

    """
    if shp is None:
        raise ValueError('shp cannot be None')

    # Parse shapefile
    sf = shapefile.Reader(shp=shp, dbf=dbf)

    # This list will be used to store the shapes
    shapes = []

    for idx, shape in enumerate(sf.shapes()):
        points = shape.points

        parts = []
        part_indices = shape.parts
        if part_indices and len(part_indices) > 0:
            # Extract each part
            part_indices.append(len(points))
            for first, last in zip(part_indices[:-1], part_indices[1:]):
                parts.append([list(x) for x in points[first:last]])

        # Form a record for this shape
        shape_record = { 'parts': parts }

        # Do we have data associated with the shape?
        if dbf is not None:
            # FIXME: This assumes a _lot_ about the field format
            data = { }
            for k, v in zip(sf.fields[1:], sf.record(idx)):
                data[str(k[0])] = v
            shape_record['data'] = data

        shapes.append(shape_record)

    return shapes

def desbribe_shp_file_collection(shp, dbf, prj):
    """Form a dictionary describing the joint contents of a shp, dbf and prj file.

    *shp*, *dbf* and *prj* should be file-like objects for reading the
    appropriate data or *None* if the data is not available. If *shp* is None,
    a *ValueError* is raised.

    """
    if shp is None:
        raise ValueError('shp cannot be None')

    rv = { 'shapes': describe_shapefile(shp, dbf) }

    if prj is not None:
        prj_contents = prj.read()
        srs = osr.SpatialReference()
        if srs.ImportFromWkt(prj_contents) != 0:
            raise RuntimeError('GDAL failed to parse projection as Well Known Text format projection.')
        srs.Fixup()
        srs.AutoIdentifyEPSG()
        rv['projection'] = srs.ExportToWkt()

    return rv

def main():
    # Specify the command line arguments and parse them
    parser = argparse.ArgumentParser(description='convert a SHP file to a KML documents')
    parser.add_argument('shp', metavar='FILENAME.shp', type=str, nargs=1,
            help='a SHP file to parse')
    parser.add_argument('--no-search', dest='search', action='store_false', default=True,
            help='do not search for matching DBF/PRJ files')
    args = parser.parse_args()

    # Open the associated files
    shp_path = args.shp[0]
    log.info('Reading shape information from %s.' % (shp_path,))
    shp = open(shp_path)

    if args.search:
        log.info('Searching for related files.')
        base, _ = os.path.splitext(shp_path)

        # Look for DBF file
        dbf_path = None
        for ext in ['.DBF', '.dbf']:
            if os.path.exists(base + ext):
                dbf_path = base + ext

        # Look for PRJ file
        prj_path = None
        for ext in ['.PRJ', '.prj']:
            if os.path.exists(base + ext):
                prj_path = base + ext

    if dbf_path is not None:
        log.info('Reading record information from %s.' % (dbf_path,))
        dbf = open(dbf_path)
    else:
        log.info('No .DBF file found: no record information will be saved.')
        dbf = None

    if prj_path is not None:
        log.info('Reading projection information from %s.' % (prj_path,))
        prj = open(prj_path)
    else:
        log.info('No .PRJ file found: no projection information will be saved.')
        prj = None

    # Convert to a dict
    d = desbribe_shp_file_collection(shp, dbf, prj)

    print(anyjson.serialize(d))
