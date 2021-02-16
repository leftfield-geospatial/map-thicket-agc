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

from collections import OrderedDict
import geopandas as gpd, pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import linear_model
from scipy import stats as stats
from map_thicket_agc import imaging as img
from map_thicket_agc import feature_selection as fs
from map_thicket_agc import calibration as calib
from map_thicket_agc import visualisation as vis
from map_thicket_agc import get_logger
from scripts import root_path
import joblib

image_root_path = root_path.joinpath(r'data/inputs/imagery')
sampling_plot_gt_file = root_path.joinpath(r'data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp')
calib_plot_file = root_path.joinpath(r'data/inputs/geospatial/gef_calib_plots.shp')
calib_plot_out_file = root_path.joinpath(r'data/outputs/geospatial/gef_calib_plots_with_agc.geojson')

image_files_dict = {'WV3 Oct 2017': image_root_path.joinpath(r'WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif'),
               'WV3 Nov 2018': image_root_path.joinpath(r'WorldView3_Nov2018_OrthoThinSpline_NoAtcor_PanSharpMs.tif'),
               'WV3 Aug 2017': image_root_path.joinpath(r'WorldView3_Aug2017_OrthoThinSpline_NoAtcor_PanSharpMs.tif'),
               'NGI April 2015': image_root_path.joinpath(r'Ngi_May2015_OrthoNgiDem_Xcalib_Rgbn_Mosaic.vrt')}

feats_of_interest = ['log(mean(R/pan))', 'log(mean(G/R))', 'log(mean(R/NIR))', '(mean(NDVI))', '(mean(SAVI))', 'log(mean(B/R))']

logger = get_logger(__name__)
logger.info('Starting...')

sampling_plot_agc_gdf = gpd.GeoDataFrame.from_file(sampling_plot_gt_file)
sampling_plot_agc_gdf.index = sampling_plot_agc_gdf['ID']

calib_plot_gdf = gpd.GeoDataFrame.from_file(calib_plot_file)

im_sampling_plot_agc_gdf_dict = {}
im_calib_plot_gdf_dict = {}

## extract features from images into geodataframes
for image_key, image_file in image_files_dict.items():
    fex = img.MsImageFeatureExtractor(image_file, plot_data_gdf=sampling_plot_agc_gdf)
    im_sampling_plot_agc_gdf = fex.extract_image_features(feat_keys=feats_of_interest)
    del fex

    if calib_plot_file == sampling_plot_gt_file:
        im_calib_plot_gdf = im_sampling_plot_agc_gdf
    else:
        fex = img.MsImageFeatureExtractor(image_file, plot_data_gdf=calib_plot_gdf, store_thumbnail=False)
        im_calib_plot_gdf = fex.extract_image_features(feat_keys=feats_of_interest)     # limit the features to save time
        del fex

    # calculate versions of ABC and AGC normalised by actual polygon area, rather than theoretical plot sizes, and append to im_sampling_plot_agc_gdf
    carbon_polynorm_dict = {}
    for plot_id, plot in im_sampling_plot_agc_gdf['data'].iterrows():
        if 'Abc' in plot and 'LitterCHa' in plot:
            litter_c_ha = np.max([plot['LitterCHa'], 0.])
            abc = np.max([plot['Abc'], 0.])
            abc_ha = abc * (100. ** 2) / plot['geometry'].area
            carbon_polynorm_dict[plot_id] = {'AbcHa2': abc_ha, 'AgcHa2': litter_c_ha + abc_ha}

    carbon_polynorm_df = pd.DataFrame.from_dict(carbon_polynorm_dict, orient='index')

    for key in ['AbcHa2', 'AgcHa2']:
        im_sampling_plot_agc_gdf[('data', key)] = carbon_polynorm_df[key]

    # fix stratum labels
    im_sampling_plot_agc_gdf.loc[im_sampling_plot_agc_gdf['data']['Stratum'] == 'Degraded', ('data', 'Stratum')] = 'Severe'

    im_sampling_plot_agc_gdf_dict[image_key] = im_sampling_plot_agc_gdf
    im_calib_plot_gdf_dict[image_key] = im_calib_plot_gdf

if True:    # write out a flattened calib and test WV3 Oct 2017 geodataframes with feat and AGC vals for use in GEE
    gdf = pd.concat([im_calib_plot_gdf_dict['WV3 Oct 2017']['data'], im_calib_plot_gdf_dict['WV3 Oct 2017']['feats']],
                    axis=1) # flatten for export
    model_filename = root_path.joinpath(r'data/outputs/Models/best_univariate_model_py38_cv5v2.joblib')
    model, model_feat_keys, model_scores = joblib.load(model_filename)
    gdf['AgcHa'] = model.predict(gdf[model_feat_keys])      # model est val, not the allometric val
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(calib_plot_out_file, driver='GeoJSON')

## find the best features for AGC modelling for each image
image_feat_scores = OrderedDict()
for image_key, im_sampling_plot_agc_gdf in im_sampling_plot_agc_gdf_dict.items():
    feat_scores = OrderedDict()
    for feat_key, feat_df in im_sampling_plot_agc_gdf['feats'][feats_of_interest].iteritems():
        scores, predicted = fs.score_model(feat_df.to_numpy().reshape(-1, 1), im_sampling_plot_agc_gdf['data', 'AgcHa'],
                                           model=linear_model.LinearRegression(), cv=5, find_predicted=True)
        feat_scores[feat_key] = {'-RMSE': scores['test_-RMSE'].mean(), 'R2': scores['R2_stacked']}
    image_feat_scores[image_key] = feat_scores
    logger.info(f'{image_key}:')
    logger.info('\n' + pd.DataFrame.from_dict(feat_scores, orient='index').sort_values(by='R2', ascending=False).to_string())

## find correlation of (select) features between images
image_feat_corr = OrderedDict()
feats_of_interest = ['log(mean(R/pan))', 'log(mean(G/R))', 'log(mean(R/NIR))', '(mean(NDVI))', '(mean(SAVI))', 'log(mean(B/R))']
for image_i, (image_key, im_sampling_plot_agc_gdf) in enumerate(im_calib_plot_gdf_dict.items()):
    if image_key == 'WV3 Oct 2017':
        ref_im_sampling_plot_agc_gdf = im_sampling_plot_agc_gdf
        continue
    r2 = np.zeros(len(im_sampling_plot_agc_gdf_dict))
    feat_corr = OrderedDict()
    for feat_i, feat_key in enumerate(feats_of_interest):
        ref_feat = ref_im_sampling_plot_agc_gdf['feats'][feat_key]
        feat = im_sampling_plot_agc_gdf['feats'][feat_key]
        (slope, intercept, r2, p, stde) = stats.linregress(ref_feat, feat)
        feat_corr[feat_key] = r2
        # print(f'slope: {slope}, intercept: {intercept}')
        if False:
            pyplot.figure(feat_i)
            pyplot.subplot(2, 2, image_i)
            xlabel = f'WV3 Oct 2017 - {feat_key}'
            ylabel = f'{image_key} - {feat_key}'
            vis.scatter_ds(pd.DataFrame(data=np.array([ref_feat, feat]).transpose(), columns=[xlabel, ylabel]))
    image_feat_corr[f'+{image_key}'] = feat_corr

image_feat_corr_df = pd.DataFrame.from_dict(image_feat_corr)
logger.info('Correlation of features between WV3 Oct 2017 and...')
logger.info('\n' + image_feat_corr_df.to_string())
logger.info('Average correlation of features over images')
logger.info('\n' + image_feat_corr_df.mean(axis=1).to_string())
logger.info('Average correlation of features over images')
logger.info('\n' + image_feat_corr_df.mean(axis=1).to_string())

## run the temporal calibration accuracy test with univariate model and log(mean(R/pan) feature
calib_feat_keys = ['log(mean(R/pan))']
model_data_dict = {}
for image_key, im_sampling_plot_agc_gdf in im_sampling_plot_agc_gdf_dict.items():
    model_data_dict[image_key]  = im_sampling_plot_agc_gdf['feats'][calib_feat_keys]

calib_data_dict = {}
for image_key, im_calib_plot_gdf in im_calib_plot_gdf_dict.items():
    calib_data_dict[image_key] = im_calib_plot_gdf['feats'][calib_feat_keys]

y = im_sampling_plot_agc_gdf_dict['WV3 Oct 2017']['data']['AgcHa'] / 1000
calib_strata = im_calib_plot_gdf_dict['WV3 Oct 2017']['data']['Stratum']

eval_calib = calib.EvaluateCalibration(model_data_dict=model_data_dict, y=y, calib_strata=calib_strata,
                                       calib_data_dict=calib_data_dict, model=linear_model.LinearRegression)

model_scores, calib_scores = eval_calib.test(n_bootstraps=100, n_calib_plots=30)
# eval_calib.print_scores()

logger.info('Done\n')
if __name__ =='__main__':
    input('Press ENTER to continue...')
