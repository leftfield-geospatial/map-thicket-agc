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

## extract features from multi-spectral satellite image
plot_agc_shapefile_name = root_path.joinpath(r'data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp')
image_filename = root_path.joinpath(r'data/inputs/imagery/WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif')

logger = get_logger(__name__)
logger.info('Starting...')

plot_agc_gdf = gpd.GeoDataFrame.from_file(plot_agc_shapefile_name)
plot_agc_gdf = plot_agc_gdf.set_index('ID').sort_index()

fex = img.MsImageFeatureExtractor(image_filename=image_filename, plot_data_gdf=plot_agc_gdf)
im_plot_agc_gdf = fex.extract_image_features()

# calculate versions of ABC and AGC normalised by actual polygon area, rather than theoretical plot sizes
carbon_polynorm_dict = {}
for plot_id, plot in im_plot_agc_gdf['data'].iterrows():
    if 'Abc' in plot and 'LitterCHa' in plot:
        litter_c_ha = np.max([plot['LitterCHa'], 0.])
        abc = np.max([plot['Abc'], 0.])
        abc_ha = abc * (100. ** 2) / plot['geometry'].area
        carbon_polynorm_dict[plot_id] = {'AbcHa2': abc_ha, 'AgcHa2': litter_c_ha + abc_ha}

carbon_polynorm_df = pd.DataFrame.from_dict(carbon_polynorm_dict, orient='index')

for key in ['AbcHa2', 'AgcHa2']:    # append to im_plot_data_gdf
    im_plot_agc_gdf[('data', key)] = carbon_polynorm_df[key]

# fix stratum labels
im_plot_agc_gdf.loc[im_plot_agc_gdf['data']['Stratum'] == 'Degraded', ('data', 'Stratum')] = 'Severe'
im_plot_agc_gdf.loc[im_plot_agc_gdf['data']['Stratum'] == 'Intact', ('data', 'Stratum')] = 'Pristine'

# make an example scatter plot of feature vs AGC/ABC
pyplot.rcParams["font.family"] = "arial"
pyplot.rcParams["font.size"] = "12"
pyplot.rcParams["font.style"] = "normal"
pyplot.rcParams['legend.fontsize'] = 'medium'
pyplot.rcParams['figure.titlesize'] = 'medium'

pyplot.figure()
vis.scatter_ds(im_plot_agc_gdf, x_col=('feats', 'log(mean(pan/R))'), y_col=('data', 'AgcHa'), class_col=('data', 'Stratum'),
               xfn=lambda x: x, do_regress=True)

## select and analyse best features for predicting AGC with linear regression
# TODO - experiment with cv vals in fs and eval below - has a big effect on what is selected and how it is scored.
#       eg, cv=10, selects ~45 features and gets R2~.93, cv=5 selects ~10 features with R2~.88
y = im_plot_agc_gdf['data']['AgcHa']
# selected_feats_df, selected_scores = fs.fcr(im_plot_agc_gdf['feats'], y, max_num_feats=None, dist_fn=None, score_fn=None)
cv = 5
selected_feats_df, selected_scores = fs.forward_selection(im_plot_agc_gdf['feats'], y, max_num_feats=25, cv=cv,
                                                          score_fn=None)

# calculate scores of selected features with LOOCV
selected_loocv_scores = []
num_feats = range(1, len(selected_scores)+1)
for i in num_feats:
    scores, predicted = fs.score_model(selected_feats_df.to_numpy()[:, :i], y, model=linear_model.LinearRegression(),
                                       find_predicted=True, cv=len(selected_feats_df))  #cv=len(selected_feats_df)
    loocv_scores = {'R2': scores['R2_stacked'], 'RMSE': np.abs(scores['test_-RMSE']).mean()/1000., 'RMSE CI':
        np.percentile(np.abs(scores['test_-RMSE']), [5, 95])}
    selected_loocv_scores.append(loocv_scores)
    logger.info('Scored model {0} of {1}'.format(i, len(selected_scores)))

selected_loocv_scores_df = pd.DataFrame(selected_loocv_scores)
best_model_idx = np.argmin(selected_loocv_scores_df['RMSE'])

# make plots of change in score as features are added to model for report
fig = pyplot.figure()
fig.set_size_inches(8, 6, forward=True)
pyplot.subplot(2, 1, 1)
pyplot.plot(num_feats, selected_loocv_scores_df['R2'], 'k-x')
pyplot.xlabel('Number of features')
pyplot.ylabel('$\mathit{R}^2$')
pyplot.tight_layout()
pyplot.subplot(2, 1, 2)
pyplot.plot(num_feats, selected_loocv_scores_df['RMSE'], 'k-x')
pyplot.xlabel('Number of features')
pyplot.ylabel('RMSE (t C ha$^{-1}$)')
pyplot.tight_layout()
pyplot.pause(.2)
pyplot.savefig(root_path.joinpath(f'data/outputs/plots/agc_acc_vs_num_feats1b_py38_cv{cv}.png'), dpi=300)

fig, ax1 = pyplot.subplots()
fig.set_size_inches(10, 4, forward=True)
color = 'tab:red'
ax1.plot(num_feats, selected_loocv_scores_df['R2'], color=color, marker='x')
ax1.set_xlabel(r'Number of features')
ax1.set_ylabel(r'$\mathit{R}^2$', color=color)  # we already handled the x-label with ax1
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(num_feats, -selected_loocv_scores_df['RMSE'], color=color, marker='o')
ax2.set_ylabel(r'-RMSE (t C ha$^{-1}$)', color=color)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_xticks(num_feats)
ax2.axis([num_feats[0]-1, num_feats[-1]+1, None, None])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax2_lim = ax2.axis()
ax2.plot([best_model_idx + 1, best_model_idx + 1], [ax2_lim[2], ax2_lim[3]], 'k--', label='Selected multivariate model')
# pyplot.text(best_model_idx + 1.5, ax2_lim[2] + (ax2_lim[3]-ax2_lim[2])/3, 'Selected multivariate model',
#             fontdict={'size': 9, 'weight': 'normal'})
pyplot.legend(loc='upper left')
ax2.axis(ax2_lim)
pyplot.pause(.2)
fig.savefig(root_path.joinpath(f'data/outputs/plots/agc_acc_vs_num_feats2b_py38_cv{cv}.png'), dpi=300)

## fit best multivariate model based on selected features
logger.info('Multivariate scores:')
best_model_idx = np.argmin(selected_loocv_scores_df['RMSE'])
scores_mv, predicted_mv = fs.score_model(selected_feats_df.iloc[:, :best_model_idx + 1], y/1000, model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=selected_feats_df.shape[0], print_scores=True)    #
logger.info('Multivariate features:')
logger.info(selected_feats_df.columns[:best_model_idx+1].to_numpy())

f = pyplot.figure()
f.set_size_inches(5, 4, forward=True)
vis.scatter_y_actual_vs_pred(y/1000., predicted_mv, scores_mv)
pyplot.savefig(root_path.joinpath(r'data/outputs/Plots/meas_vs_pred_agc_multivariate_model_b.png'), dpi=300)

# fitting
best_multivariate_model = linear_model.LinearRegression()
best_multivariate_model.fit(selected_feats_df.iloc[:, :best_model_idx+1], y/1000)
logger.info('Multivariate coefficients:')
np.set_printoptions(precision=4, suppress=True)
logger.info(np.array(best_multivariate_model.coef_))
logger.info('Multivariate intercept:')
logger.info(np.array(best_multivariate_model.intercept_))

# save model
joblib.dump([best_multivariate_model, selected_feats_df.columns[:best_model_idx+1].to_numpy(), scores_mv], root_path.joinpath(f'data/outputs/Models/best_multivariate_model_py38_cv{cv}v2.joblib'))
pickle.dump([best_multivariate_model, selected_feats_df.columns[:best_model_idx+1].to_numpy(), scores_mv], open(root_path.joinpath(f'data/outputs/Models/best_multivariate_model_py38_cv{cv}v2.pickle'), 'wb'))

# make feature table for JARS paper
mv_model_feat_dict = dict(Features=selected_feats_df.columns[:best_model_idx+1].to_numpy(),
                          Coefficients=best_multivariate_model.coef_)
mv_model_feat_df = pd.DataFrame.from_dict(mv_model_feat_dict).round(decimals=4)
mv_model_feat_df.index += 1
# mv_model_feat_df.to_clipboard()

## fit best univariate model based on selected feature
logger.info('Univariate model scores:')
scores_uv, predicted_uv = fs.score_model(selected_feats_df.iloc[:, :1], y/1000, model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=selected_feats_df.shape[0], print_scores=True)

logger.info('Univariate model features:')
logger.info(selected_feats_df.columns[:1].to_numpy())

f = pyplot.figure()
f.set_size_inches(5, 4, forward=True)
vis.scatter_y_actual_vs_pred(y/1000., predicted_uv, scores_uv)
pyplot.savefig(root_path.joinpath(r'data/outputs/Plots/meas_vs_pred_agc_univariate_model_b.png'), dpi=300)

f = pyplot.figure()
f.set_size_inches(10, 4, forward=True)
pyplot.subplot(1, 2, 1)
pyplot.title('(a)')
vis.scatter_y_actual_vs_pred(y/1000., predicted_mv, scores_mv)
pyplot.subplot(1, 2, 2)
pyplot.title('(b)')
vis.scatter_y_actual_vs_pred(y/1000., predicted_uv, scores_uv)
pyplot.savefig(root_path.joinpath(r'data/outputs/Plots/meas_vs_pred_agc_both_models_b.png'), dpi=300)

# fitting
best_univariate_model = linear_model.LinearRegression(fit_intercept=True)
best_univariate_model.fit(selected_feats_df.iloc[:, :1], y/1000)
logger.info('Univariate model coefficient:')
logger.info(np.array(best_univariate_model.coef_))
logger.info('Univariate model intercept:')
logger.info(np.array(best_univariate_model.intercept_))

# save model
joblib.dump([best_univariate_model, selected_feats_df.columns[:1].to_numpy(), scores_uv], root_path.joinpath(f'data/outputs/Models/best_univariate_model_py38_cv{cv}v2.joblib'))
pickle.dump([best_univariate_model, selected_feats_df.columns[:1].to_numpy(), scores_uv], open(root_path.joinpath(f'data/outputs/Models/best_univariate_model_py38_cv{cv}v2.pickle'), 'wb'))

# make accuracy table for JARS paper
acc_dict = dict()
for key, N, scores in zip(['Multivariate', 'Univariate'], [best_model_idx+1, 1], [scores_mv, scores_uv]):
    rmse_ptile = np.percentile(np.abs(scores['test_-RMSE']), [5, 95])
    rmse_ptile_str = f'{rmse_ptile[0]:.4f} - {rmse_ptile[1]:.4f}'
    rrmse = 100*np.abs(scores['test_-RMSE']).mean() / np.mean(y / 1000)
    acc_dict[key] = {'N': int(N), 'R2': float(scores['R2_stacked']), 'RMSE': float(np.abs(scores['test_-RMSE']).mean()),
                            'RMSE 5-95%': rmse_ptile_str, 'RRMSE': float(rrmse)}

acc_df = pd.DataFrame.from_dict(acc_dict).T
acc_df[['R2','RMSE','RRMSE']] = np.around(acc_df[['R2','RMSE','RRMSE']].to_numpy(dtype=float), decimals=4)
# acc_df.to_clipboard()

## Analyse outliers for JARS paper
if True:
    from sklearn.neighbors import KernelDensity
    # add error and max tree height rank to dataframe
    err_mv = predicted_mv - y/1000
    im_plot_agc_gdf.loc[:, ('data','predicted_mv')] = predicted_mv
    im_plot_agc_gdf.loc[:, ('data','err_mv')] = err_mv

    arg_max_height_sort = np.argsort(-im_plot_agc_gdf[('data', 'MaxHeight')])
    max_height_rank = np.empty_like(arg_max_height_sort)
    max_height_rank[arg_max_height_sort] = np.arange(len(arg_max_height_sort))
    im_plot_agc_gdf.loc[:, ('data', 'max_height_rank')] = max_height_rank

    # predicted vs actual AGC with plot labels and thumbnails
    pyplot.figure()
    vis.scatter_ds(im_plot_agc_gdf, x_col=('data', 'AgcHa'), y_col=('data', 'predicted_mv'),
                   xfn=lambda x: x/1000, do_regress=True, label_col=('data', 'ID'), thumbnail_col=('data', 'thumbnail'),
                   x_label='Measured AGC', y_label='Predicted AGC (mv model)')
    pyplot.title('Actual vs predicted AGC')

    # investigate correlation between signed error and maximum height in plot
    pyplot.figure()
    vis.scatter_ds(im_plot_agc_gdf, x_col=('data', 'err_mv'), y_col=('data', 'MaxHeight'),
                   xfn=lambda x: x, do_regress=True, label_col=('data', 'ID'), thumbnail_col=('data', 'thumbnail'),
                   x_label='Signed mv model error', y_label='Max. height')
    pyplot.title('Correlation between signed error and max height')

    # investigate correlation between abs(error) and maximum height in plot
    pyplot.figure()
    vis.scatter_ds(im_plot_agc_gdf, x_col=('data', 'err_mv'), y_col=('data', 'MaxHeight'),
                   xfn=lambda x: np.abs(x), do_regress=True, label_col=('data', 'ID'), thumbnail_col=('data', 'thumbnail'),
                   x_label='Abs(mv model error)', y_label='Max. height')
    pyplot.title('Correlation between abs error and max height')

    # investigate correlation between abs(error) and actual AGC
    pyplot.figure()
    vis.scatter_ds(im_plot_agc_gdf, x_col=('data', 'err_mv'), y_col=('data', 'AgcHa'), xfn=lambda x: np.abs(x),
                   yfn=lambda x: x/1000, do_regress=True, label_col=('data', 'ID'), thumbnail_col=('data', 'thumbnail'),
                   x_label='Abs(mv model error)', y_label='Measured AGC')
    pyplot.title('Correlation between abs error and AGC')

    # investigate correlation between abs(error) and actual AGC for Intact stratum
    im_plot_agc_gdf_intact = im_plot_agc_gdf.loc[im_plot_agc_gdf[('data','Stratum')]=='Pristine']
    pyplot.figure()
    vis.scatter_ds(im_plot_agc_gdf_intact, x_col=('data', 'err_mv'), y_col=('data', 'AgcHa'), xfn=lambda x: np.abs(x),
                   yfn=lambda x: x/1000, do_regress=True, label_col=('data', 'ID'), thumbnail_col=('data', 'thumbnail'),
                   x_label='Abs(mv model error)', y_label='Actual AGC')
    pyplot.title('Correlation between abs error and AGC for intact stratum')

    # investigate the distribution of errors
    kde = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(err_mv.to_numpy().reshape(-1,1))
    err_plot = np.linspace(-25, 25, 1000)[:, np.newaxis]
    err_mv_kd = kde.score_samples(err_plot)
    pyplot.figure()
    pyplot.plot(err_plot, np.exp(err_mv_kd))
    pyplot.plot(err_mv, np.zeros_like(err_mv), 'kx')
    pyplot.title('Signed error distribution')

    # error stats per stratum
    for stratum, stratum_df in im_plot_agc_gdf.groupby(by=('data', 'Stratum')):
        mean_err = np.abs(stratum_df.loc[:, ('data', 'err_mv')]).mean()
        std_err = np.abs(stratum_df.loc[:, ('data', 'err_mv')]).std()/np.sqrt(len(stratum_df))
        print(f'{stratum} mean(se) error: {mean_err:.4f}({std_err:.4f})')


    # which plots were the worst performers, and is there a connection with max tree height?
    print('First 10 lowest performing plots:')
    err_mv_sort = err_mv.abs().sort_values(ascending=False)[:10]
    print(err_mv_sort)

    print('MaxHeights and corresp ranks of 10 lowest underestimated plots:')
    print(im_plot_agc_gdf.loc[err_mv_sort.index, ('data', ['MaxHeight','max_height_rank'])])


logger.info('Done\n')
if __name__ =='__main__':
    input('Press ENTER to continue...')

