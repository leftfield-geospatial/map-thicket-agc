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
##
from scripts import root_path
from agc_estimation import imaging as img
import joblib
import time
from agc_estimation import get_logger
logger = get_logger(__name__)

logger.info('Starting...')

image_filename = r"D:/OneDrive/GEF Essentials/Source Images/WorldView3 Oct 2017/WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif"

if True:
    map_filename = root_path.joinpath(r'data/outputs/geospatial/gef5_slm_wv3_oct_2017_univariate_agc_10m_w33s33.tif')
    model_filename = root_path.joinpath(r'data/outputs/Models/best_univariate_model_py38_cv5v2.joblib')
else:
    map_filename = root_path.joinpath(r'data/outputs/geospatial/gef5_slm_wv3_oct_2017_multivariate_agc_10m_w33s33.tif')
    model_filename = root_path.joinpath(r'data/outputs/Models/best_multivariate_model_py38_cv5v2.joblib')

## load model and apply to image
model, model_feat_keys, model_scores = joblib.load(model_filename)
mapper = img.MsImageMapper(image_file_name=image_filename, map_file_name=map_filename, model=model, model_feat_keys=model_feat_keys,
                         save_feats=False)
start = time.time()
mapper.map(win_size=(33, 33), step_size=(33, 33))   # map with ~10m pixels
logger.info(f'Mapping duration: {(time.time()-start):.2f}s')
img.thicket_agc_post_proc(mapper)                   # remove noise and place sensible limits on AGC

logger.info('Done\n')
input('Press ENTER to continue...')