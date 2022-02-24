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
from scipy import stats as stats

import joblib, pickle

from map_thicket_agc import imaging as img
from map_thicket_agc import visualisation as vis
from map_thicket_agc import feature_selection as fs
from map_thicket_agc import get_logger
import pathlib
## extract features from multi-spectral satellite image
plot_agc_shapefile_name = root_path.joinpath(r'data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp')
# source_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Reference\COPERNICUS-S2-20151023T081112_20151023T081949_T34HGH_B4328.tif")
# source_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Reference\LANDSAT-LC08-C02-T1_L2-LC08_172083_20150525_B4325.tif")
source_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Source\NGI_3323DA_2015_GefSite_Source.vrt")

# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Reference\COPERNICUS-S2-20151023T081112_20151023T081949_T34HGH_B4328.tif")
homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k5_5.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\Landsat\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k5_5.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k7_7.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k11_11.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-BLK-OFFSET_k1_1.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN_k5_5.vrt")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\COPERNICUS-S2-20151023T081112_20151023T081949_T34HGH_B4328_HOMO_cREF_mGAIN-BLK-OFFSET_k1_1.tif")
# homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN-OFFSET_k15_15.vrt")
homo_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Homogenised\NGI_3323D_2015_GefSite_RGBN_HOMO_cREF_mGAIN_k15_15.vrt")

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
pyplot.rcParams["font.family"] = "arial"
pyplot.rcParams["font.size"] = "12"
pyplot.rcParams["font.style"] = "normal"
pyplot.rcParams['legend.fontsize'] = 'medium'
pyplot.rcParams['figure.titlesize'] = 'medium'

pyplot.figure()
vis.scatter_ds(homo_gdf, x_col=('feats', 'log(mean(NDVI))'), y_col=('data', 'AgcHa'), class_col=('data', 'Stratum'),
               xfn=lambda x: x, do_regress=True, label_col=('data', 'ID'))


def plot_agc_corr(x, y, x_label='NDVI', y_label='AGC (t C ha$^{-1}$)'):
    xlim = [np.nanmin(x), np.nanmax(x)]
    ylim = [np.nanmin(y), np.nanmax(y)]
    xd = np.diff(xlim)[0]
    yd = np.diff(ylim)[0]

    pyplot.axis('tight')
    pyplot.axis(xlim + ylim)
    pyplot.plot(x, y, marker='.', linestyle='None', markersize=5)

    if True:
        (slope, intercept, r, p, stde) = stats.linregress(x, y)
        pyplot.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.3f}'.format(r ** 2),
                    fontdict={'size': 12})
        yr = np.array(xlim) * slope + intercept
        pyplot.plot(xlim, yr, 'k--', lw=2, zorder=-1)

    pyplot.xlabel(x_label, fontdict={'size': 12})
    pyplot.ylabel(y_label, fontdict={'size': 12})
    # pyplot.axis('tight')


labels = ['Source', 'Homogenised']
fig = pyplot.figure()
fig.set_size_inches(10, 4.5, forward=True)

for i, gdf in enumerate([source_gdf, homo_gdf]):
    pyplot.subplot(1, 2, i+1)
    plot_agc_corr(gdf[('feats', '(mean(NDVI))')], gdf[('data', 'AgcHa')]/1000)
    pyplot.title(labels[i])


