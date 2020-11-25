"""
  GEF5-SLM: Above ground carbon estimation in thicket using multi-spectral images
  Copyright (C) 2020 Dugal Harris
  Released under GNU Affero General Public License (AGPL) (https://www.gnu.org/licenses/agpl.html)
  Email: dugalh@gmail.com
"""

import numpy as np
import geopandas as gpd, pandas as pd
import pathlib, sys, os
from sklearn import linear_model
from matplotlib import pyplot
import logging
import joblib, pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())

sys.path.append(str(root_path.joinpath('agc_estimation')))
logging.basicConfig(format='%(levelname)s %(name)s: %(message)s')

from agc_estimation import imaging as img
from agc_estimation import visualisation as vis
from agc_estimation import feature_selection as fs

#--------------------------------------------------------------------------------------------------------------
# WV3 im analysis
plot_agc_shapefile_name = root_path.joinpath(r'data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp')
image_filename = r"D:/OneDrive/GEF Essentials/Source Images/WorldView3 Oct 2017/WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif"

plot_agc_gdf = gpd.GeoDataFrame.from_file(plot_agc_shapefile_name)
plot_agc_gdf = plot_agc_gdf.set_index('ID').sort_index()

fex = img.MsImageFeatureExtractor(image_filename=image_filename, plot_data_gdf=plot_agc_gdf)
im_plot_agc_gdf = fex.extract_image_features()
    # im_plot_data_gdf.pop('ST49')

# calculate versions of ABC and AGC normalised by actual polygon area, rather than theoretical plot sizes, and append to im_plot_data_gdf
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

# make some scatter plots of features vs AGC/ABC
pyplot.figure()
# vis.scatter_ds(im_plot_data_gdf, x_col=('feats', 'pan/R'), y_col=('data', 'AgcHa'), class_col=('data', 'Stratum'),
#                xfn=lambda x: np.log10(x), do_regress=True)
vis.scatter_ds(im_plot_agc_gdf, x_col=('feats', '(mean(pan/R))'), y_col=('data', 'AgcHa'), class_col=('data', 'Stratum'),
               xfn=lambda x: np.log10(x), do_regress=True)
pyplot.figure()
vis.scatter_ds(im_plot_agc_gdf, x_col=('feats', 'sqr(mean(R/G))'), y_col=('data', 'AbcHa'), class_col=('data', 'Stratum'),
               xfn=lambda x: np.log10(x), do_regress=True)
pyplot.figure()
vis.scatter_ds(im_plot_agc_gdf, x_col=('feats', '(mean(pan/R))'), y_col=('data', 'AbcHa'), class_col=('data', 'Stratum'),
               xfn=lambda x: np.log10(x), do_regress=True, thumbnail_col=('data','thumbnail'), label_col=('data', 'ID'))

# select best features for predicting AGC with linear regression
# TODO - experiment with different cv vals here and below - it has a big effect on what is selected and how it is scored.
#  Likewise, so do small numerical differences in feats.
y = im_plot_agc_gdf['data']['AgcHa']
selected_feats_df, selected_scores =  fs.forward_selection(im_plot_agc_gdf['feats'], y, max_num_feats=25, cv=5,  #cv=X.shape[0] / 5
                                                           score_fn=None)
# feat_scores = fs.ranking(im_plot_data_gdf['feats'], y, cv=5, score_fn=None)

# calculate scores of selected features with LOOCV
selected_loocv_scores = []
num_feats = range(0, len(selected_scores))
for i in num_feats:
    scores, predicted = fs.score_model(selected_feats_df.to_numpy()[:, :i + 1], y, model=linear_model.LinearRegression(), find_predicted=True, cv=len(selected_feats_df))
    loocv_scores = {'R2': scores['R2_stacked'], 'RMSE': np.abs(scores['test_-RMSE']).mean()/1000., 'RMSE CI': np.percentile(np.abs(scores['test_-RMSE']), [5, 95])}
    selected_loocv_scores.append(loocv_scores)
    print('Scored model {0} of {1}'.format(i+1, len(selected_scores)))

selected_loocv_scores_df = pd.DataFrame(selected_loocv_scores)

# make plots of change in score as features are added to model for report
fig = pyplot.figure()
fig.set_size_inches(8, 6, forward=True)
pyplot.subplot(2, 1, 1)
pyplot.plot(num_feats, selected_loocv_scores_df['R2'], 'k-')
pyplot.xlabel('Number of features')
pyplot.ylabel('$/mathit{R}^2$')
pyplot.tight_layout()
pyplot.subplot(2, 1, 2)
pyplot.plot(num_feats, selected_loocv_scores_df['RMSE'], 'k-')
pyplot.xlabel('Number of features')
pyplot.ylabel('RMSE (t C ha$^{-1}$)')
pyplot.tight_layout()
pyplot.pause(.1)
pyplot.savefig(root_path.joinpath(r'data/outputs/plots/agc_acc_vs_num_feats1b_py38_cv10.png'), dpi=300)

fig, ax1 = pyplot.subplots()
fig.set_size_inches(8, 4, forward=True)
color = 'tab:red'
ax1.set_xlabel('Number of features')
ax1.set_ylabel('$/mathit{R}^2$', color=color)  # we already handled the x-label with ax1
ax1.plot(num_feats, selected_loocv_scores_df['R2'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('-RMSE (t C ha$^{-1}$)', color=color)  # we already handled the x-label with ax1
ax2.plot(num_feats, -selected_loocv_scores_df['RMSE'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
pyplot.pause(.1)
fig.savefig(root_path.joinpath(r'data/outputs/agc_acc_vs_num_feats2b_py38_cv10.png'), dpi=300)

#------------------------------------------------------------------------------------------------------------------------
# Fit best multiple and single feature models, generate acccuracy stats and plots
# multiple feat model
logger.info('Multi feat model scores:')
best_model_idx = np.argmin(selected_loocv_scores_df['RMSE'])
scores, predicted = fs.score_model(selected_feats_df.iloc[:, :best_model_idx + 1], y/1000, model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=selected_feats_df.shape[0], print_scores=True)
logger.info('Multi feat model features:')
logger.info(selected_feats_df.columns[:best_model_idx+1].to_numpy())

fig = pyplot.figure()
fig.set_size_inches(5, 4, forward=True)
vis.scatter_y_actual_vs_pred(y/1000., predicted, scores)
fig.savefig(root_path.joinpath(r'data/outputs/Plots/meas_vs_pred_agc_multivariate_model_b.png'), dpi=300)

best_multivariate_model = linear_model.LinearRegression()
best_multivariate_model.fit(selected_feats_df.iloc[:, :best_model_idx+1], y/1000)
logger.info('Multi feat model coefficients:')
logger.info(np.array(best_multivariate_model.coef_))
logger.info('Multi feat model intercept:')
logger.info(np.array(best_multivariate_model.intercept_))

if True:
    joblib.dump([best_multivariate_model, selected_feats_df.columns[:best_model_idx+1].to_numpy(), scores], root_path.joinpath(r'data/outputs/Models/best_multivariate_model_py38_cv5v2.joblib'))
    pickle.dump([best_multivariate_model, selected_feats_df.columns[:best_model_idx+1].to_numpy(), scores], open(root_path.joinpath(r'data/outputs/Models/best_multivariate_model_py38_cv5v2.pickle'), 'wb'))

# single feat model
logger.info('Single feat model scores:')
scores, predicted = fs.score_model(selected_feats_df.iloc[:, :1], y/1000, model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=selected_feats_df.shape[0], print_scores=True)

logger.info('Single feat model features:')
logger.info(selected_feats_df.columns[:1].to_numpy())

fig = pyplot.figure()
fig.set_size_inches(5, 4, forward=True)
vis.scatter_y_actual_vs_pred(y/1000., predicted, scores)
fig.savefig(root_path.joinpath(r'data/outputs/Plots/meas_vs_pred_agc_univariate_model_b.png'), dpi=300)

# fitting
best_univariate_model = linear_model.LinearRegression(fit_intercept=True)
best_univariate_model.fit(selected_feats_df.iloc[:, :1], y/1000)
logger.info('Single feat model coefficient:')
logger.info(np.array(best_univariate_model.coef_))
logger.info('Single feat model intercept:')
logger.info(np.array(best_univariate_model.intercept_))

if True:
    joblib.dump([best_univariate_model, selected_feats_df.columns[:1].to_numpy(), scores], root_path.joinpath(r'data/outputs/Models/best_univariate_model_py38_cv5v2.joblib'))
    pickle.dump([best_univariate_model, selected_feats_df.columns[:1].to_numpy(), scores], open(root_path.joinpath(r'data/outputs/Models/best_univariate_model_py38_cv5v2.pickle'), 'wb'))

