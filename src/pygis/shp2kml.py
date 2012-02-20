import argparse
import logging
import os
import pyproj
import html
import shapefile
import xml.dom
import xml.dom.minidom

log = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

kml_ns = 'http://www.opengis.net/kml/2.2'

shp_proj = pyproj.Proj(init='epsg:27700') # British national grid
kml_proj = pyproj.Proj(init='epsg:4326')  # Projection used for KML

def shape_to_placemark(doc, shape, id_str, name=None, record = None, fields = None):
    # Create the placemark element
    placemark = doc.createElement('Placemark')
    placemark.setAttribute('id', id_str)

    if name is not None:
        name_elem = doc.createElement('name')
        name_elem.appendChild(doc.createTextNode(name))
        placemark.appendChild(name_elem)

    # Create a description of the element as a html document
    desc = html.HTML()

    # Do we have records?
    if record is not None:
        table = desc.table()
        th = table.thead()
        tr = th.tr()
        tr.td('Field')
        tr.td('Value')

        tb = table.tbody()
        for k, v in zip(fields, record):
            tr = tb.tr()
            tr.td(str(k[0]))
            tr.td(str(v))

    # Append the HTML description to the placemark
    description = doc.createElement('description')
    description.appendChild(doc.createTextNode(str(desc)))
    placemark.appendChild(description)

    # Project all of the co-ordinates
    lngs, lats = pyproj.transform(shp_proj, kml_proj,
            [p[0] for p in shape.points], [p[1] for p in shape.points])

    # Create a line string for the outline
    ls = doc.createElement('LineString')
    placemark.appendChild(ls)

    # Create a co-ordinates element with all these points in it
    coords = doc.createElement('coordinates')
    for lng, lat in zip(lngs, lats):
        coords.appendChild(doc.createTextNode('%f,%f,0' % (lng, lat)))
    ls.appendChild(coords)

    return placemark

def main():
    # Specify the command line arguments and parse them
    parser = argparse.ArgumentParser(description='convert a SHP file to a KML documents')
    parser.add_argument('shp', metavar='FILENAME.shp', type=str, nargs=1,
            help='a SHP file to parse')
    parser.add_argument('-d, --dbf', metavar='FILENAME.dbf', type=str, nargs='?', dest='dbf',
            help='a database file to read associated shape records from', default=None)
    args = parser.parse_args()

    # Load SHP and (optionally) DBF file
    shp = open(args.shp[0])
    dbf = open(args.dbf) if args.dbf is not None else None

    # Create a file id based on the name of the file
    file_id, _ = os.path.splitext(os.path.basename(args.shp[0]))

    # Load the shape file
    sf = shapefile.Reader(shp=shp, dbf=dbf)
    log.info('%s: loaded %s shapes' % (args.shp[0], len(sf.shapes())))

    # Create a document to hold the KML
    dom = xml.dom.getDOMImplementation()
    kml_doc = dom.createDocument(kml_ns, 'kml', None)
    kml = kml_doc.documentElement
    kml.setAttribute('xmlns', kml_ns)

    # Create the containing Document node
    document = kml_doc.createElement('Document')
    document.setAttribute('id', file_id)
    kml.appendChild(document)

    # Set up the style
    style = kml_doc.createElement('Style')
    style.setAttribute('id', 'shape')
    document.appendChild(style)

    # Create a line style with specified width and color
    line_style = kml_doc.createElement('LineStyle')
    style.appendChild(line_style)

    width = kml_doc.createElement('width')
    width.appendChild(kml_doc.createTextNode(str(2)))
    line_style.appendChild(width)

    color = kml_doc.createElement('color')
    color.appendChild(kml_doc.createTextNode('ff0000dd'))
    line_style.appendChild(color)

    # Get a list of shapes and, if present, records. If no DBF file is
    # provided, generate a list of None records.
    shapes = sf.shapes()
    records = sf.records() if dbf is not None else ([None,] * len(shapes))
    fields = sf.fields if dbf is not None else []

    # Convert each shape in the file
    for idx, sr_pair in enumerate(zip(shapes, records)):
        shape, record = sr_pair
        id_str = 'shape_%i' % idx
        name = 'shape %i' % idx
        pm = shape_to_placemark(kml_doc, shape, id_str,
                name = name, record = record, fields = fields[1:])
        document.appendChild(pm)

        # Extend the placemark with styling
        style_url = kml_doc.createElement('styleUrl')
        style_url.appendChild(kml_doc.createTextNode('#shape'))
        pm.appendChild(style_url)

    # Output the XML
    print(kml_doc.toprettyxml(indent='  '))
