import sys
import SpatialUtils as su
import pylab
import numpy as np
from sklearn import linear_model, metrics
import joblib, pickle
import logging
# logging.basicConfig(level=logging.DEBUG)
# reload(su)


# inFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\NGI\My Rectified\3321D_319_05_0147_RGBN.tif"
# outFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\NGI\My Rectified\3321D_319_05_0147_RGBN_Test.tif"

# inFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\Digital Globe\058217622010_01\PCI Output\ATCOR\SRTM+AdjCorr Aligned Photoscan DEM\o17OCT01084657-M2AS_R1C12-058217622010_01_P001_PanAndPansharpMS_gdalv3.vrt"
inFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\Digital Globe\058217622010_01\PCI Output\ATCOR\SRTM+AdjCorr Aligned Photoscan DEM\o17OCT01084657-M2AS_R1C12-058217622010_01_P001_PanAndPansharpMS.tif"

# inFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\Digital Globe\058217622010_01\PCI Output\ATCOR\SRTM+AdjCorr Aligned Photoscan DEM\ATCORCorrected_o17OCT01084657-P2AS_R1C12-058217622010_01_P001_PhotoscanDEM_14128022_PanSharp.pix"
# inFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\Digital Globe\058217622010_01\PCI Output\ATCOR\SRTM+AdjCorr\ATCORCorrected_o17OCT01084657_R1C12-058217622010_01_P001_14368043_PanSharpen.pix"
# inFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\Digital Globe\058217622010_01\PCI Output\ATCOR\Basic\ATCORCorrected_o17OCT01084657_R1C12-058217622010_01_P001_14368025_PanSharp.tif"

outFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\Digital Globe\058217622010_01\PCI Output\ATCOR\SRTM+AdjCorr Aligned Photoscan DEM\o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py38Cv5v1_9Feat_10m.tif"
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModel.joblib'
modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModel.pickle'
modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModelPy38Cv5v1.pickle'
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestSingleTermModel.joblib'
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModelPy27NoTxt.joblib'
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModelPy27Cv5v2.joblib'
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModelPy27Cv5NoCv2.joblib'
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModelPy27Cv5v2.joblib'
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestSingleTermModelPy27Cv5NoCv2.joblib'

# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestSingleTermModel.joblib'

# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestSingleTermModel.pickle'
# modelFile = r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModel.pickle'
# gdal command lines (gdal v3+ to work with qgis)
# build pan and pansharp vrt with nodata
# gdalbuildvrt -separate -srcnodata 0 -vrtnodata 0 test.vrt ATCORCorrected_o17OCT01084657-P2AS_R1C12-058217622010_01_P001_PhotoscanDEM_14128022.pix ATCORCorrected_o17OCT01084657-P2AS_R1C12-058217622010_01_P001_PhotoscanDEM_14128022_PanSharp.pix
# convert vrt to tiff (pcidisk driver has a bug)
# gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=DEFLATE" -co "BIGTIFF=YES" -co NUM_THREADS=ALL_CPUS -a_nodata 0 o17OCT01084657-M2AS_R1C12-058217622010_01_P001_PanAndPansharpMS_gdalv3.vrt o17OCT01084657-M2AS_R1C12-058217622010_01_P001_PanAndPansharpMS.tif
# gdaladdo -ro -r average --config COMPRESS_OVERVIEW DEFLATE --config PHOTOMETRIC_OVERVIEW RGB --config INTERLEAVE_OVERVIEW BAND o17OCT01084657-M2AS_R1C12-058217622010_01_P001_PanAndPansharpMS.tif 2 4 8 16 32 64 128

# lm, selected_keys = joblib.load(modelFile)
lm, selected_keys = pickle.load(open(modelFile, 'rb'), encoding='latin1')
# def RollingWindow(a, window, step_size=1):
# 	shape = a.shape[:-1] + (1 + (a.shape[-1] - window)/step_size, window)
# 	strides = a.strides[:-1] + (step_size*a.strides[-1], a.strides[-1])
# 	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)
#
# #
# #
# a = np.arange(0, 30).reshape(3, 10)
# window = 3
# step_size = 2
# RollingWindow(a, window, step_size=step_size).shape
if sys.version_info.major == 3 and type(selected_keys[0]) is bytes:
    selected_keys = [sk.decode() for sk in selected_keys]       # convert bytes to unicode for py 3
print([(i,key) for i,key in enumerate(selected_keys)])
# print(zip(np.arange(1,len(selected_keys)+1), selected_keys))


mapper = su.AgcMap(in_file_name=inFile, out_file_name=outFile, model=lm, model_keys=selected_keys,
                   feat_ex_fn=su.ImPlotFeatureExtractor.extract_patch_ms_features_ex, save_feats=False)

mapper.Create(win_size=(33, 33), step_size=(33, 33))
mapper.PostProc()
# mapper.Exec(win_size=(1, 1), step_size=(1, 1))

# gdal_fillnodata -md 20 -si 0 -b 1 o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m.tif -of GTiff -co COMPRESS=DEFLATE o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m_fillnodata.tif
# gdal_merge -o GEF_WV3_Oct2017_AGC_10m.tif -co COMPRESS=DEFLATE -separate -a_nodata nan o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_1Feat_10m_postproc.tif o17OCT01084657-M2AS_R1C12-058217622010_01_P001_AGC_Py27Cv5v2_24Feat_10m_postproc.tif

