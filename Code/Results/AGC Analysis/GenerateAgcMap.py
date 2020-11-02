import sys
from modules import modelling as su
import numpy as np
import pickle

# logging.basicConfig(level=logging.DEBUG)
# reload(su)


inFile = r"D:\OneDrive\GEF Essentials\Source Images\WorldView3 Oct 2017\WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif"

outFile = r"C:\Data\Development\Projects\GEF-5 SLM\Data\Outputs\Geospatial\GEF5 SLM - WV3 Oct 2017 - Univariate AGC - 10m.tif"
modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestSingleTermModel.pickle'
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModelPy38Cv5v1.pickle'

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

mapper = su.ApplyLinearModel(in_file_name=inFile, out_file_name=outFile, model=lm, model_keys=selected_keys,
                             feat_ex_fn=su.ImageFeatureExtractor.extract_patch_ms_features_ex, save_feats=False)

mapper.create(win_size=(33, 33), step_size=(33, 33))
mapper.post_proc()

# gdal_fillnodata -md 20 -si 0 -b 1 o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m.tif -of GTiff -co COMPRESS=DEFLATE o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m_fillnodata.tif
# gdal_merge -o GEF_WV3_Oct2017_AGC_10m.tif -co COMPRESS=DEFLATE -separate -a_nodata nan o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_1Feat_10m_postproc.tif o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m_postproc.tif

