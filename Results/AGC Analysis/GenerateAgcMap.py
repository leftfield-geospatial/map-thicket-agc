#
   """
      GEF5-SLM Above ground carbon estimation in thicket using multi-spectral images
      Copyright (C) 2020 Dugal Harris
      Released under GNU Affero General Public License (AGPL) (https://www.gnu.org/licenses/agpl.html)
      email dugalh@gmail.com
   """




#

      GEF5-SLM Above ground carbon estimation in thicket using multi-spectral images
      Copyright (C) 2020  Dugal Harris

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU Affero General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU Affero General Public License for more details.

      You should have received a copy of the GNU Affero General Public License
      along with this program.  If not, see <https://www.gnu.org/licenses/>.



import sys
from agc_estimation import imaging as su
import numpy as np
import pickle

# logging.basicConfig(level=logging.DEBUG)
# reload(su)


inFile = r"D:/OneDrive/GEF Essentials/Source Images/WorldView3 Oct 2017/WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif"

outFile = r"C:/Data/Development/Projects/GEF-5 SLM/Data/outputs/geospatial/GEF5 SLM - WV3 Oct 2017 - Univariate AGC - 10m.tif"
modelFile = r'C:/Data/Development/Projects/PhD GeoInformatics/Docs/Funding/GEF5/Invoices, Timesheets and Reports/Final Report/bestSingleTermModel.pickle'
# modelFile = r'C:\data\Development\Projects\PhD GeoInformatics\docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\best_model_py38_cv5v1.pickle'

# gdal command lines (gdal v3+ to work with qgis)
# build pan and pansharp vrt with nodata
# gdalbuildvrt -separate -srcnodata 0 -vrtnodata 0 test.vrt ATCORCorrected_o17OCT01084657-P2AS_R1C12-058217622010_01_P001_PhotoscanDEM_14128022.pix ATCORCorrected_o17OCT01084657-P2AS_R1C12-058217622010_01_P001_PhotoscanDEM_14128022_PanSharp.pix
# convert vrt to tiff (pcidisk driver has a bug)
# gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=DEFLATE" -co "BIGTIFF=YES" -co NUM_THREADS=ALL_CPUS -a_nodata 0 o17OCT01084657-M2AS_R1C12-058217622010_01_P001_PanAndPansharpMS_gdalv3.vrt o17OCT01084657-M2AS_R1C12-058217622010_01_P001_PanAndPansharpMS.tif
# gdaladdo -ro -r average --config COMPRESS_OVERVIEW DEFLATE --config PHOTOMETRIC_OVERVIEW RGB --config INTERLEAVE_OVERVIEW BAND o17OCT01084657-M2AS_R1C12-058217622010_01_P001_PanAndPansharpMS.tif 2 4 8 16 32 64 128

# lm, selected_keys = joblib.load(modelFile)
lm, selected_keys = pickle.load(open(modelFile, 'rb'), encoding='latin1')

if sys.version_info.major == 3 and (type(selected_keys[0]) is bytes or type(selected_keys[0]) is np.bytes_):
    selected_keys = [sk.decode() for sk in selected_keys]       # convert bytes to unicode for py 3
print([(i,key) for i,key in enumerate(selected_keys)])

mapper = su.ImageMapper(image_file_name=inFile, map_file_name=outFile, model=lm, model_feat_keys=selected_keys,
                        feat_ex_fn=su.MsImageFeatureExtractor.extract_patch_ms_features_ex, save_feats=False)

mapper.map(win_size=(33, 33), step_size=(33, 33))
mapper._post_proc()

# gdal_fillnodata -md 20 -si 0 -b 1 o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m.tif -of GTiff -co COMPRESS=DEFLATE o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m_fillnodata.tif
# gdal_merge -o GEF_WV3_Oct2017_AGC_10m.tif -co COMPRESS=DEFLATE -separate -a_nodata nan o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_1Feat_10m_postproc.tif o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m_postproc.tif

