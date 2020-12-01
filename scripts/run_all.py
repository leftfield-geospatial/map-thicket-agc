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

# TODO revise necessity for logging path code below in installed code
import logging, pathlib, sys, os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())

sys.path.append(str(root_path))
logging.basicConfig(format='%(levelname)s %(name)s: %(message)s')

# run all scripts in the right order
from scripts import generate_agc_ground_truth
from scripts import generate_agc_shapefile
from scripts import fit_agc_model
from scripts import generate_agc_map
from scripts import evaluate_calibration
