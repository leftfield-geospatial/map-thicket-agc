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
import matplotlib
matplotlib.use("TkAgg")
matplotlib.interactive(True)
import pathlib, sys, os
import logging
from collections import OrderedDict
import geopandas as gpd, pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import linear_model
from scipy import stats as stats
from agc_estimation import imaging as img
from agc_estimation import feature_selection as fs
from agc_estimation import calibration as calib
from agc_estimation import visualisation as vis

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())

sys.path.append(str(root_path))
logging.basicConfig(format='%(levelname)s %(name)s: %(message)s')

image_root_path = pathlib.Path(r"D:/OneDrive/GEF Essentials/Source Images")
sampling_plot_gt_file = root_path.joinpath(r"data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp")

image_files_dict = {'WV3 Oct 2017': image_root_path.joinpath(r"WorldView3 Oct 2017/WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif"),
               'WV3 Nov 2018': image_root_path.joinpath(r"WorldView3 Nov 2018/WorldView3_Nov2018_OrthoThinSpline_NoAtcor_PanSharpMs.tif"),
               'WV3 Aug 2017': image_root_path.joinpath(r"WorldView3 Aug 2017/WorldView3_Aug2017_OrthoThinSpline_NoAtcor_PanSharpMs.tif"),
               'NGI April 2015': image_root_path.joinpath(r"NGI April 2015/Ngi_May2015_OrthoNgiDem_Xcalib_Rgbn_Mosaic.vrt")}

plot_agc_gdf = gpd.GeoDataFrame.from_file(sampling_plot_gt_file)
im_plot_agc_gdf_dict = {}

# extract features from images into geodataframes
for image_key, image_file in image_files_dict.items():
    fex = img.MsImageFeatureExtractor(image_file, plot_data_gdf=plot_agc_gdf)
    im_plot_agc_gdf = fex.extract_image_features()
    del(fex)

    # calculate versions of ABC and AGC normalised by actual polygon area, rather than theoretical plot sizes, and append to im_plot_agc_gdf
    carbon_polynorm_dict = {}
    for plot_id, plot in im_plot_agc_gdf['data'].iterrows():
        if 'Abc' in plot and 'LitterCHa' in plot:
            litter_c_ha = np.max([plot['LitterCHa'], 0.])
            abc = np.max([plot['Abc'], 0.])
            abc_ha = abc * (100. ** 2) / plot['geometry'].area
            carbon_polynorm_dict[plot_id] = {'AbcHa2': abc_ha, 'AgcHa2': litter_c_ha + abc_ha}

    carbon_polynorm_df = pd.DataFrame.from_dict(carbon_polynorm_dict, orient='index')

    for key in ['AbcHa2', 'AgcHa2']:
        im_plot_agc_gdf[('data', key)] = carbon_polynorm_df[key]

    # fix stratum labels
    im_plot_agc_gdf.loc[im_plot_agc_gdf['data']['Stratum'] == 'Degraded', ('data', 'Stratum')] = 'Severe'
    im_plot_agc_gdf.loc[im_plot_agc_gdf['data']['Stratum'] == 'Intact', ('data', 'Stratum')] = 'Pristine'

    im_plot_agc_gdf_dict[image_key] = im_plot_agc_gdf


# find the best features for AGC modelling for each image
image_feat_scores = OrderedDict()
feats_of_interest = ['log(mean(R/pan))', 'log(mean(G/R))', 'log(mean(R/NIR))', '(mean(NDVI))', '(mean(SAVI))', 'log(mean(B/R))']
for image_key, im_plot_agc_gdf in im_plot_agc_gdf_dict.items():
    feat_scores = OrderedDict()
    for feat_key, feat_df in im_plot_agc_gdf['feats'][feats_of_interest].iteritems():
        scores, predicted = fs.score_model(feat_df.to_numpy().reshape(-1, 1), im_plot_agc_gdf['data', 'AgcHa'],
                                           model=linear_model.LinearRegression(), cv=5, find_predicted=True)
        feat_scores[feat_key] = {'-RMSE': scores['test_-RMSE'].mean(), 'R2': scores['R2_stacked']}
    image_feat_scores[image_key] = feat_scores
    print(f'{image_key}:')
    print(pd.DataFrame.from_dict(feat_scores, orient='index').sort_values(by='R2', ascending=False),'\n')

# find correlation of (select) features between images
image_feat_corr = OrderedDict()
feats_of_interest = ['log(mean(R/pan))', 'log(mean(G/R))', 'log(mean(R/NIR))', '(mean(NDVI))', '(mean(SAVI))', 'log(mean(B/R))']
for image_i, (image_key, im_plot_agc_gdf) in enumerate(im_plot_agc_gdf_dict.items()):
    if image_key == 'WV3 Oct 2017':
        ref_im_plot_agc_gdf = im_plot_agc_gdf
        continue
    r2 = np.zeros(len(im_plot_agc_gdf_dict))
    feat_corr = OrderedDict()
    for feat_i, feat_key in enumerate(feats_of_interest):
        ref_feat = ref_im_plot_agc_gdf['feats'][feat_key]
        feat = im_plot_agc_gdf['feats'][feat_key]
        (slope, intercept, r2, p, stde) = stats.linregress(ref_feat, feat)
        feat_corr[feat_key] = r2
        if False:
            pyplot.figure(feat_i)
            pyplot.subplot(2, 2, image_i)
            xlabel = f'WV3 Oct 2017 - {feat_key}'
            ylabel = f'{image_key} - {feat_key}'
            vis.scatter_ds(pd.DataFrame(data=np.array([ref_feat, feat]).transpose(), columns=[xlabel, ylabel]))
    image_feat_corr[f'+{image_key}'] = feat_corr

image_feat_corr_df = pd.DataFrame.from_dict(image_feat_corr)
print('Correlation of features between WV3 Oct 2017 and...')
print(image_feat_corr_df)
print('Average correlation of features over images')
print(image_feat_corr_df.mean(axis=1))

# run the temporal calibration accuracy test with univariate model and log(mean(R/pan) feature
calib_feat_keys = ['log(mean(R/pan))']
model_data_dict = {}
for image_key, im_plot_agc_gdf in im_plot_agc_gdf_dict.items():
    model_data_dict[image_key]  = im_plot_agc_gdf['feats'][calib_feat_keys]
y = im_plot_agc_gdf_dict['WV3 Oct 2017']['data']['AgcHa'] / 1000
strata = im_plot_agc_gdf_dict['WV3 Oct 2017']['data']['Stratum']

eval_calib = calib.EvaluateCalibration(model_data_dict=model_data_dict, y=y, strata=strata,
                                       calib_data_dict=model_data_dict, model=linear_model.LinearRegression)

model_scores, calib_scores = eval_calib.test(n_bootstraps=100, n_calib_plots=8)
eval_calib.print_scores()
