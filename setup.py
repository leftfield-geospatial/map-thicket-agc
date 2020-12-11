"""
    GEF5-SLM: Above ground carbon estimation in thicket using multi-spectral images
    Copyright (C) 2020 Dugal Harris
    Email: dugalh@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from setuptools import setup, find_packages
import glob

# To install local development version use:
#    pip install -e .

setup(
    name='map-thicket-agc',
    version='0.1.0',
    description='Mapping AGC in thicket with multi-spectral imagery',
    author='Dugal Harris',
    author_email='dugalh@gmail.com',
    url='https://github.com/dugalh/map_thicket_agc/blob/master/setup-py',
    license='AGPLv3',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.5',
        'matplotlib>=3.3',
        'openpyxl>=3.0',
        'geopandas>=0.8',
        'rasterio>=1.1',
        'scikit-learn>=0.23'
    ],
    # scripts=glob.glob('scripts/*.py'),
    # data_files=[('map_thicket_agc_data', glob.glob('data/inputs/**/*.*', recursive=True))],
)
