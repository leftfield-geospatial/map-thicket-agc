"""
  GEF5-SLM Above ground carbon estimation in thicket using multi-spectral images
  Copyright (C) 2020 Dugal Harris
  Released under GNU Affero General Public License (AGPL) (https://www.gnu.org/licenses/agpl.html)
  email dugalh@gmail.com
"""
import pathlib, sys, os
import logging
import geopandas as gpd, pandas as pd
import numpy as np
from sklearn import linear_model
from agc_estimation import imaging as img
from agc_estimation import calibration as calib

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())

sys.path.append(str(root_path.joinpath('agc_estimation')))
logging.basicConfig(format='%(levelname)s %(name)s: %(message)s')

image_root_path = pathlib.Path(r"D:/OneDrive/GEF Essentials/Source Images")
sampling_plot_gt_file = root_path.joinpath(r"data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp")

image_files = [image_root_path.joinpath(r"WorldView3 Oct 2017/WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif"),
               image_root_path.joinpath(r"WorldView3 Nov 2018/WorldView3_Nov2018_OrthoThinSpline_NoAtcor_PanSharpMs.tif"),
               image_root_path.joinpath(r"WorldView3 Aug 2017/WorldView3_Aug2017_OrthoThinSpline_NoAtcor_PanSharpMs.tif"),
               image_root_path.joinpath(r"NGI April 2015/Ngi_May2015_OrthoNgiDem_Xcalib_Rgbn_Mosaic.vrt")]


plot_agc_gdf = gpd.GeoDataFrame.from_file(sampling_plot_gt_file)
im_plot_agc_gdfs = []

# extract features from images into geodataframes
for image_file in image_files:
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

    im_plot_agc_gdfs.append(im_plot_agc_gdf)

# find the best single features for Wv3 2017 AGC modelling

# check correlation of features between 2018 WV3 and other images to see which are best


if False:   # find best single feat models so that we know which feats to try calibrate with
    X, y, feat_keys = feature_extractors[0].get_feat_array_ex(y_data_key='AgcHa')
    r2 = []
    rmse = []
    from collections import OrderedDict
    univariate_model_scores = OrderedDict()
    for ki, key in enumerate(feat_keys):
        xv = X[:, ki]
        scores, predicted = su.feature_selection.score_model( X[:, ki].reshape(-1, 1), y / 1000.,
                                                           model=linear_model.LinearRegression(),
                                                           find_predicted=True, cv=10, print_scores=False)
        univariate_model_scores[key] = {'r2': scores['R2_stacked'], 'rmse': -scores['test_user'].mean()}
        print('.', end=' ')

    r2 = np.array([sfms['r2'] for sfms in list(univariate_model_scores.values())])
    rmse = np.array([sfms['rmse'] for sfms in list(univariate_model_scores.values())])
    idx = np.argsort(-r2)
    print('\nWV2017 feats sorted by best single feat model')
    print(list(zip(feat_keys[idx[:50]], r2[idx[:50]])))

# plots for report
if False:
    # based on above results, these are keys of interest i.e. good single term models that are not too redundant and are common between NGI and WV3
    keys_of_interest = np.array(['Log(R/pan)', 'Log(G/R)', 'Log(R/NIR)', 'NDVI', 'SAVI', 'Log(B/R)'])
    feat_scores = OrderedDict()
    fd_labels = ['WV3 2017', 'WV3 2018', 'NGI 2015']
    for ki, key in enumerate(keys_of_interest):
        feat_scores[key] = OrderedDict({'WV3 2017 AGC R2':univariate_model_scores[key]['r2']})

    for fi, feat_dict in enumerate(implot_feat_dicts[1:]):
        # keys_of_interest = np.array(['Std(pan)', 'Std(NDVI)', 'R', 'G', 'B', 'NIR', 'R/pan', 'G/pan', 'B/pan', 'NIR/pan', 'NDVI', 'SAVI', 'NIR/R', 'pan'])
        r2 = np.zeros(keys_of_interest.__len__())
        for ki, key in enumerate(keys_of_interest):
            xv = np.array([x['feats'][key] for x in list(implot_feat_dicts[0].values())])
            yv = np.array([x['feats'][key] for x in list(feat_dict.values())])
            # local_scatter_plot(xv, yv, xlabel=key+'_2017', ylabel=key+'_2018')
            (slope, intercept, r2[ki], p, stde) = stats.linregress(xv, yv)
            feat_scores[key][fd_labels[fi+1]] = r2[ki]

            # r2[ki], rmse = su.scatter_plot(xv, yv, xlabel=key+fd_labels[0], ylabel=key+fd_labels[fi+1], class_labels=None, labels=None)
        print('{0} features sorted by R2:'.format(fd_labels[fi+1]))
        sort_idx = np.argsort(-r2)
        print(list(zip(keys_of_interest[sort_idx], r2[sort_idx])))
        import pandas as pd
        df = pd.DataFrame(feat_scores).transpose()
        df['Mean'] = df.mean(axis=1)
        df.to_excel(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\temporalCalibFeats.xlsx')
        print(df)

if False:
    doPlots = True
    fd_labels = ['WV3 2017', 'WV3 2018', 'NGI 2015']
    ref_feat_keys = list(implot_feat_dicts[0].values())[0]['feats'].keys()
    for fi, feat_dict in enumerate(implot_feat_dicts[1:]):
        feat_keys = list(feat_dict.values())[0]['feats'].keys()
        if doPlots:
            pylab.figure()
        ip = 1
        keys_of_interest = np.intersect1d(ref_feat_keys, feat_keys)
        keys_of_interest = np.array(['R/pan', 'NDVI', 'R', 'NIR/pan'])
        keys_of_interest = np.array(['Log(R/pan)', 'G/R', 'Log(R/NIR)', 'NDVI'])
        labels = keys_of_interest
        r2 = np.zeros(keys_of_interest.__len__())
        for ki, key in enumerate(keys_of_interest):
            xv = np.array([x['feats'][key] for x in list(implot_feat_dicts[0].values())])
            yv = np.array([x['feats'][key] for x in list(feat_dict.values())])
            # local_scatter_plot(xv, yv, xlabel=key+'_2017', ylabel=key+'_2018')
            (slope, intercept, r2[ki], p, stde) = stats.linregress(xv, yv)
            if doPlots:
                pylab.subplot(2, 2, ip)
                xlabel = '{0} - {1}'.format(fd_labels[0], labels[ki])
                ylabel = '{0} - {1}'.format(fd_labels[fi+1], labels[ki])
                su.scatter_plot(xv, yv, xlabel=xlabel, ylabel=ylabel, class_labels=None,
                                labels=None)
                pylab.tight_layout()
            ip += 1
            # r2[ki], rmse = su.scatter_plot(xv, yv, xlabel=key+fd_labels[0], ylabel=key+fd_labels[fi+1], class_labels=None, labels=None)
        print('{0} features sorted by R2:'.format(fd_labels[fi + 1]))
        sort_idx = np.argsort(-r2)
        print(list(zip(keys_of_interest[sort_idx], r2[sort_idx])))

calib_feat_keys = ['log(mean(R/pan))']
model_data_list = []
for im_plot_agc_gdf in im_plot_agc_gdfs:
    model_data_list.append(im_plot_agc_gdf['feats'][calib_feat_keys])

eval_calib = calib.EvaluateCalibration(model_data_list=model_data_list, y=im_plot_agc_gdfs[0]['data']['AgcHa'] / 1000., strata=im_plot_agc_gdfs[0]['data']['Stratum'],
                                       calib_data_list=model_data_list, model=linear_model.LinearRegression)

model_scores, calib_scores = eval_calib.test(n_bootstraps=100, n_calib_plots=8)
eval_calib.print_scores()
