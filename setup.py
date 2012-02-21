from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()


version = '0.1'

install_requires = [
    'anyjson',
    'argparse',
    'html',
    'gdal',
    'pyshp',
    'pyproj',
]


setup(name='PyGIS',
    version=version,
    description="Python utilities for GIS files",
    long_description=README + '\n\n' + NEWS,
    classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    keywords='',
    author='Rich Wareham',
    author_email='rjw57@cantab.net',
    url='http://github.com/rjw57/pygis',
    license='APACHE-2.0',
    packages=find_packages('src'),
    package_dir = {'': 'src'},include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        'console_scripts':
            ['shp2xml=pygis.shp2kml:main', 'shp2json=pygis.shp2json:main']
    }
)
