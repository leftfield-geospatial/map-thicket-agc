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

from map_thicket_agc import root_path
import numpy as np
import geopandas as gpd, pandas as pd
from sklearn import linear_model
from matplotlib import pyplot

import joblib, pickle

from map_thicket_agc import imaging as img
from map_thicket_agc import visualisation as vis
from map_thicket_agc import feature_selection as fs
from map_thicket_agc import get_logger
import pathlib
## extract features from multi-spectral satellite image
plot_agc_shapefile_name = root_path.joinpath(r'data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp')
# image_filename = root_path.joinpath(r'data/inputs/imagery/WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif')
# image_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Source\NGI_3323DA_2015_GefSite_Source.vrt")
# source_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Reference\COPERNICUS-S2-20151023T081112_20151023T081949_T34HGH_B4328.tif")
source_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Source\NGI_3323DA_2015_GefSite_Source.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k5_5.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\Landsat\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k5_5.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k7_7.vrt")
homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k11_11.vrt")

logger = get_logger(__name__)
logger.info('Starting...')

# load ground truth
plot_agc_gdf = gpd.GeoDataFrame.from_file(plot_agc_shapefile_name)
plot_agc_gdf = plot_agc_gdf.set_index('ID').sort_index()

source_fex = img.MsImageFeatureExtractor(image_filename=source_filename, plot_data_gdf=plot_agc_gdf)
source_gdf = source_fex.extract_image_features()
homo_fex = img.MsImageFeatureExtractor(image_filename=homo_filename, plot_data_gdf=plot_agc_gdf)
homo_gdf = homo_fex.extract_image_features()

# fix stratum labels
for gdf in [source_gdf, homo_gdf]:
    gdf.loc[gdf['data']['Stratum'] == 'Degraded', ('data', 'Stratum')] = 'Severe'
    gdf.loc[gdf['data']['Stratum'] == 'Intact', ('data', 'Stratum')] = 'Pristine'

# make an example scatter plot of feature vs AGC/ABC
if False:
    pyplot.rcParams["font.family"] = "arial"
    pyplot.rcParams["font.size"] = "12"
    pyplot.rcParams["font.style"] = "normal"
    pyplot.rcParams['legend.fontsize'] = 'medium'
    pyplot.rcParams['figure.titlesize'] = 'medium'

labels = ['Source', 'Homogenised']
for label, gdf in zip(labels, [source_gdf, homo_gdf]):
    pyplot.figure()
    vis.scatter_ds(gdf, x_col=('feats', 'log(mean(pan/R))'), y_col=('data', 'AgcHa'), class_col=('data', 'Stratum'),
                   xfn=lambda x: x, do_regress=True)
    pyplot.title(label)
    pyplot.figure()
    vis.scatter_ds(gdf, x_col=('feats', 'log(mean(NDVI))'), y_col=('data', 'AgcHa'), class_col=('data', 'Stratum'),
                   xfn=lambda x: x, do_regress=True)
    pyplot.title(label)


# mask = ~np.isnan(im_plot_agc_gdf['feats']['log(mean(pan/R))'])
# im_plot_agc_gdf = im_plot_agc_gdf.loc[mask]
#
# pyplot.figure()
# pyplot.plot(im_plot_agc_gdf[('feats', 'log(mean(pan/R))')][mask], im_plot_agc_gdf[('data', 'AgcHa')][mask] , '.')
# np.corrcoef(im_plot_agc_gdf[('feats', 'log(mean(pan/R))')][mask],  im_plot_agc_gdf[('data', 'AgcHa')][mask])**2
# from scipy import stats as stats
# (slope, intercept, r, p, stde) = stats.linregress(im_plot_agc_gdf[('feats', 'log(mean(pan/R))')][mask], im_plot_agc_gdf[('data', 'AgcHa')][mask])
