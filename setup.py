from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README')).read()
NEWS = open(os.path.join(here, 'NEWS.rst')).read()

from release_info import version

install_requires = [
#    'anyjson',
#    'argparse',
#    'html',
#    'pyshp',
#    'pyproj',
    'gdal',
    'numpy',
    'scipy',
    'pyopencl',
]

setup(name='pygis',
    version=version,
    description="Python utilities for GIS files",
    long_description=README + '\n\n' + NEWS,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Utilities',
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    keywords='',
    author='Rich Wareham',
    author_email='rjw57@cantab.net',
    url='http://github.com/rjw57/pygis',
    license='APACHE-2.0',
    packages=['pygis',],  #find_packages(os.path.join(here, 'src')),
    package_dir = {'': 'src'},
    package_data = {'pygis': ['data/*']},
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
#        'console_scripts':
#            ['shp2kml=pygis.shp2kml:main', 'shp2json=pygis.shp2json:main']
    }
)
