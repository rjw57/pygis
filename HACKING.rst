Development setup
=================

To create a buildout,

  $ python bootstrap.py
  $ bin/buildout

Documentation HOWTO
===================

After creating the buildout, run bin/sphinxbuilder.

Release HOWTO
=============

To make a release, 

  1) Update release date/version in NEWS.txt and setup.py
  2) Run 'python setup.py sdist'
  3) Test the generated source distribution in dist/
  4) Upload to PyPI: 'python setup.py sdist register upload'
  5) Increase version in setup.py (for next release)

