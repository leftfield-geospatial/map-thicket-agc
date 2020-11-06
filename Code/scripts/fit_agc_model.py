# use SpatialUtils to produce models and plots for report

import rasterio
import numpy as np
import geopandas as gpd, pandas as pd
import pathlib, sys, os, glob, warnings
from sklearn import linear_model
from modules import modelling as mdl
from matplotlib import pyplot
import logging
import joblib, pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[2]
else:
    root_path = pathlib.Path(os.getcwd()).parents[0]

sys.path.append(str(root_path.joinpath('Code')))
logging.basicConfig(format='%(levelname)s %(name)s: %(message)s')

#--------------------------------------------------------------------------------------------------------------
# WV3 im analysis
plot_agc_shapefile_name = root_path.joinpath(r'Data\Outputs\Geospatial\GEF Plot Polygons with AGC v2.shp')
image_filename = r"D:\OneDrive\GEF Essentials\Source Images\WorldView3 Oct 2017\WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif"

plot_agc_gdf = gpd.GeoDataFrame.from_file(plot_agc_shapefile_name)

with rasterio.open(image_filename, 'r') as imr:
    fex = mdl.ImageFeatureExtractor(image_reader=imr, plot_data_gdf=plot_agc_gdf)
    im_plot_data_gdf = fex.extract_all_features(patch_fn=mdl.ImageFeatureExtractor.extract_patch_ms_features_ex)
    # im_plot_data_gdf.pop('ST49')

# fix stratum labels
im_plot_data_gdf.loc[im_plot_data_gdf['data']['Stratum'] == 'Degraded', ('data', 'Stratum')] = 'Severe'
im_plot_data_gdf.loc[im_plot_data_gdf['data']['Stratum'] == 'Intact', ('data', 'Stratum')] = 'Pristine'


# make some scatter plots of features vs AGC/ABC
pyplot.figure()
# mdl.scatter_ds(im_plot_data_gdf, x_col=('feats', 'pan/R'), y_col=('data', 'AgcHa'), class_col=('data', 'Stratum'),
#                xfn=lambda x: np.log10(x), do_regress=True)
mdl.scatter_ds(im_plot_data_gdf, x_col=('feats', '(mean(pan/R))'), y_col=('data', 'AgcHa'), class_col=('data', 'Stratum'),
               xfn=lambda x: np.log10(x), do_regress=True)
pyplot.figure()
mdl.scatter_ds(im_plot_data_gdf, x_col=('feats', 'sqr(mean(R/G))'), y_col=('data', 'AbcHa'), class_col=('data', 'Stratum'),
               xfn=lambda x: np.log10(x), do_regress=True)
pyplot.figure()
mdl.scatter_ds(im_plot_data_gdf, x_col=('feats', '(mean(pan/R))'), y_col=('data', 'AbcHa'), class_col=('data', 'Stratum'),
               xfn=lambda x: np.log10(x), do_regress=True, thumbnail_col=('data','thumbnail'), label_col=('data', 'ID'))

# select best features for predicting AGC with linear regression
# TODO - experiment with different cv vals here and below - it has a big effect on what is selected and how it is scored.  Likewise, so do small numerical differences in feats.
y = im_plot_data_gdf['data']['AgcHa']
selected_feats_df, selected_scores =  mdl.FeatureSelector.forward_selection(im_plot_data_gdf['feats'], y, max_num_feats=25, cv=5,  #cv=X.shape[0] / 5
                                                                                        score_fn=None)
# feat_scores = mdl.FeatureSelector.ranking(im_plot_data_gdf['feats'], y, cv=5, score_fn=None)

# calculate scores of selected features with LOOCV
selected_loocv_scores = []
num_feats = range(0, len(selected_scores))
for i in num_feats:
    scores, predicted = mdl.FeatureSelector.score_model(selected_feats_df.to_numpy()[:, :i + 1], y, model=linear_model.LinearRegression(), find_predicted=True, cv=len(selected_feats_df))
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
pyplot.ylabel('$\mathit{R}^2$')
pyplot.tight_layout()
pyplot.subplot(2, 1, 2)
pyplot.plot(num_feats, selected_loocv_scores_df['RMSE'], 'k-')
pyplot.xlabel('Number of features')
pyplot.ylabel('RMSE (t C ha$^{-1}$)')
pyplot.tight_layout()
pyplot.pause(.1)
pyplot.savefig(root_path.joinpath(r'Data\Outputs\Plots\AgcAccVsNumFeats1b_Py38Cv10.png'), dpi=300)

fig, ax1 = pyplot.subplots()
fig.set_size_inches(8, 4, forward=True)
color = 'tab:red'
ax1.set_xlabel('Number of features')
ax1.set_ylabel('$\mathit{R}^2$', color=color)  # we already handled the x-label with ax1
ax1.plot(num_feats, selected_loocv_scores_df['R2'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('-RMSE (t C ha$^{-1}$)', color=color)  # we already handled the x-label with ax1
ax2.plot(num_feats, -selected_loocv_scores_df['RMSE'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
pyplot.pause(.1)
fig.savefig(root_path.joinpath(r'Data\Outputs\Plots\AgcAccVsNumFeats2b_Py38Cv10.png'), dpi=300)

#------------------------------------------------------------------------------------------------------------------------
# Fit best multiple and single feature models, generate acccuracy stats and plots
# multiple feat model
logger.info('Multi feat model scores:')
best_model_idx = np.argmin(selected_loocv_scores_df['RMSE'])
scores, predicted = mdl.FeatureSelector.score_model(selected_feats_df.iloc[:, :best_model_idx + 1], y/1000, model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=selected_feats_df.shape[0], print_scores=True)
logger.info('Multi feat model features:')
logger.info(selected_feats_df.columns[:best_model_idx+1].to_numpy())

fig = pyplot.figure()
fig.set_size_inches(5, 4, forward=True)
mdl.scatter_y_actual_vs_pred(y/1000., predicted, scores)
fig.savefig(root_path.joinpath(r'Data\Outputs\Plots\MeasVsPredAgcMultiFeatModel_b.png'), dpi=300)

# fitting
best_multi_feat_model = linear_model.LinearRegression()
best_multi_feat_model.fit(selected_feats_df.iloc[:, :best_model_idx+1], y/1000)
logger.info('Multi feat model coefficients:')
logger.info(np.array(best_multi_feat_model.coef_))
logger.info('Multi feat model intercept:')
logger.info(np.array(best_multi_feat_model.intercept_))

if True:
    joblib.dump([best_multi_feat_model, selected_feats_df.columns[:best_model_idx+1].to_numpy(), scores], root_path.joinpath(r'Data\Outputs\Models\BestMultiFeatModelPy38Cv5v2.joblib'))
    pickle.dump([best_multi_feat_model, selected_feats_df.columns[:best_model_idx+1].to_numpy(), scores], open(root_path.joinpath(r'Data\Outputs\Models\BestMultiFeatModelPy38Cv5v2.pickle'), 'wb'))

# single feat model
logger.info('Single feat model scores:')
scores, predicted = mdl.FeatureSelector.score_model(selected_feats_df.iloc[:, :1], y/1000, model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=selected_feats_df.shape[0], print_scores=True)

logger.info('Single feat model features:')
logger.info(selected_feats_df.columns[:1].to_numpy())

fig = pyplot.figure()
fig.set_size_inches(5, 4, forward=True)
mdl.scatter_y_actual_vs_pred(y/1000., predicted, scores)
fig.savefig(root_path.joinpath(r'Data\Outputs\Plots\MeasVsPredAgcSingleFeatModel_b.png'), dpi=300)

# fitting
best_single_feat_model = linear_model.LinearRegression(fit_intercept=True)
best_single_feat_model.fit(selected_feats_df.iloc[:, :1], y/1000)
logger.info('Single feat model coefficient:')
logger.info(np.array(best_single_feat_model.coef_))
logger.info('Single feat model intercept:')
logger.info(np.array(best_single_feat_model.intercept_))

if True:
    joblib.dump([best_single_feat_model, selected_feats_df.columns[:1].to_numpy(), scores], root_path.joinpath(r'Data\Outputs\Models\BestSingleFeatModelPy38Cv5v2.joblib'))
    pickle.dump([best_single_feat_model, selected_feats_df.columns[:1].to_numpy(), scores], open(root_path.joinpath(r'Data\Outputs\Models\BestSingleFeatModelPy38Cv5v2.pickle'), 'wb'))



lm = linear_model.LinearRegression()
lm.fit(Xselected_feats[:, :best_model_idx + 1], old_div(y, 1000))
joblib.dump([lm, selected_keys[:best_model_idx + 1]],
            r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModelPy38Cv5v1.joblib')
pickle.dump([lm, selected_keys[:best_model_idx + 1]], open(
    r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestModelPy38Cv5v1.pickle',
    'wb'))

lm = linear_model.LinearRegression()
lm.fit(Xselected_feats[:, :1], old_div(y, 1000))
joblib.dump([lm, selected_keys[:1]],
            r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestSingleTermModelPy38Cv5v1.joblib')
pickle.dump([lm, selected_keys[:1]], open(
    r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\bestSingleTermModelPy38Cv5v1.pickle',
    'wb'))

# TODO: - check why CV RMSE is better than RMSE on full training set,
#  x - why does FS perform worse in py 3.8 vs 2.7
#  x - entropy vs nanentropy
#  - some model / feature selection comparisons
#  - can we get around duplication feature ex fn in ApplyLinearModel?
#  - docs, boilerplate, license





if False:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn import pipeline

    pl = pipeline.make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=1.))
    scores, predicted = mdl.FeatureSelector.score_model(Xselected_feats, old_div(y, 1000), model=pl,
                                                        find_predicted=True, cv=X.shape[0], print_scores=True)

    scores, predicted = mdl.FeatureSelector.score_model(Xselected_feats[:, :best_model_idx + 1], old_div(y, 1000), model=KernelRidge(kernel='rbf', alpha=.1), find_predicted=True, cv=X.shape[0], print_scores=True)

    scores, predicted = mdl.FeatureSelector.score_model(Xs, y, model=SVR(kernel='linear', C=200), cv=10)

    scores, predicted = mdl.FeatureSelector.score_model(Xselected_feats_s, y, model=SVR(kernel='linear', C=1000000, gamma='auto'), cv=10)


# su.scatter_plot(y/1000., predicted/1000., labels=implot_feat_dict.keys())
fig = pyplot.figure()
fig.set_size_inches(5, 4, forward=True)
scatter_y_pred(y/1000., predicted, scores)
fig.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\MeasVsPredAgcBestModel.png', dpi=300)

if False:
    fig = pyplot.figure()
    mdl.scatter_plot(y / 1000., predicted, labels=list(implot_feat_dict.keys()))

fig = pyplot.figure()
fig.set_size_inches(10, 4, forward=True)
pyplot.subplot(1,2,1)
scatter_y_pred(y/1000., predicted, scores)
pyplot.title('(a)')
pyplot.subplot(1,2,2)
print('\nBest single feature model scores:')
scores, predicted = mdl.FeatureSelector.score_model(Xselected_feats[:, :1], y / 1000., model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=X.shape[0], print_scores=True)
scatter_y_pred(y/1000., predicted, scores)
pyplot.title('(b)')
fig.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\MeasVsPredAgcBestModels.png', dpi=300)

#------------------------------------------------------------------------------------------------------------------------
# Correlation analysis of the ground cover classification
import modules.modelling as mdl
import pyplot
import numpy as np
from sklearn import linear_model, metrics
reload(mdl)

plot_agc_shapefile_name = "C:/Data/Development/Projects/PhD GeoInformatics/Data/GEF Sampling/GEF Plot Polygons with Agc v5.shp"
# imageFile = r"D:/Data/Development/Projects/PhD GeoInformatics/Data/Digital Globe/058217622010_01/PCI Output/ATCOR/SRTM+AdjCorr Aligned Photoscan DEM/ATCORCorrected_o17OCT01084657-P2AS_R1C12-058217622010_01_P001_PhotoscanDEM_14128022_PanSharp.pix"
clf_file = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\NGI\GEF DEM\DSM Working\ground_clf2.tif"

vr = mdl.GdalVectorReader(plot_agc_shapefile_name)
ld = vr.read()
imr_clf = mdl.GdalImageReader(clf_file)
fex_clf = mdl.ImageFeatureExtractor(image_reader=imr_clf, plot_feat_dict=ld['GEF Plot Polygons with Agc v5'])
implot_feat_dict_clf = fex_clf.extract_all_features(patch_fn=mdl.ImageFeatureExtractor.extract_patch_clf_features)

# set DegrClass field in implot_feat_dict using plot ID
for f in list(implot_feat_dict_clf.values()):
    id = f['ID']
    if id[0] == 'S' or id[:3] == 'TCH':
        f['DegrClass'] = 'Severe'
    elif id[0] == 'M':
        f['DegrClass'] = 'Moderate'
    elif id[0] == 'P' or id[:3] == 'INT':
        f['DegrClass'] = 'Pristine'
    else:
        f['DegrClass'] = '?'

X_clf, y_clf, feat_keys_clf = fex_clf.get_feat_array_ex(y_data_key='AgcHa')
feat_scores = mdl.FeatureSelector.ranking(X_clf, y_clf, feat_keys=feat_keys_clf)
classes = [plot['DegrClass'] for plot in list(implot_feat_dict_clf.values())]

pyplot.figure()
fex_clf.scatter_plot(x_feat_key='VegCover', y_feat_key='AgcHa', do_regress=True, class_key='DegrClass', show_labels=False, yfn= lambda x: x/1000.)
pyplot.xlabel('Veg. cover (%)')
pyplot.ylabel('AGC (tC/ha)')
pyplot.tight_layout()

# feature selection and model plot
Xselected_feats, selected_scores, selected_keys = mdl.FeatureSelector.forward_selection(X_clf, y_clf, feat_keys=feat_keys_clf, max_num_feats=4, cv=5,
                                                                                        score_fn = lambda y, pred: -np.sqrt(metrics.mean_squared_error(y, pred)))
scores, predicted = mdl.FeatureSelector.score_model(Xselected_feats[:, :np.argmax(selected_scores) + 1], y_clf, model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=X_clf.shape[0], print_scores=True)

scatter_y_pred(y_clf/1000., predicted/1000., scores)



# ----------------------------------------------------------------------------------------------------------------------
# NGI image analysis
plot_agc_shapefile_name = "C:/Data/Development/Projects/PhD GeoInformatics/Data/GEF Sampling/GEF Plot Polygons with Agc v5.shp"
# imageFile = r"V:/Data/NGI/Rectified/3323D_2015_1001/RGBN/XCALIB/o3323d_2015_1001_GEF_RGBN_XCALIB.vrt"  # ""V:/Data/NGI/Rectified/3323D_2015_1001/RGBN/o3323d_2015_1001_02_0077_Lo25Wgs84_RGBN_XCALIB.tif"
image_filename = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\NGI\Rectified\3322D_2015_1001\RGBN\XCALIB\AutoGcpWv3\o3323D_2015_1001_GEF_RGBN_XCALIb_v2.vrt"
# imageFile = r"D:\Data\Development\Projects\PhD GeoInformatics\Data\NGI\Rectified\3322D_2015_1001\RGBN\AutoGcpWv3\3323d_2015_OrthoRect.vrt"

reload(mdl)

vr = mdl.GdalVectorReader(plot_agc_shapefile_name)
ld = vr.read()
imr = mdl.GdalImageReader(image_filename)
fex = mdl.ImageFeatureExtractor(image_reader=imr, plot_feat_dict=ld['GEF Plot Polygons with Agc v5'])
implot_feat_dict = fex.extract_all_features(patch_fn=mdl.ImageFeatureExtractor.extract_patch_ms_features_ex)

# set DegrClass field in implot_feat_dict using plot ID
for f in list(implot_feat_dict.values()):
    id = f['ID']
    if id[0] == 'S' or id[:3] == 'TCH':
        f['DegrClass'] = 'Severe'
    elif id[0] == 'M':
        f['DegrClass'] = 'Moderate'
    elif id[0] == 'P' or id[:3] == 'INT':
        f['DegrClass'] = 'Pristine'
    else:
        f['DegrClass'] = '?'

pyplot.figure()
fex.scatter_plot(x_feat_key='R/pan', y_feat_key='AgcHa', class_key='DegrClass', xfn=lambda x: np.log10(x))
pyplot.xlabel('R/pan')
pyplot.ylabel('AGC (tC/ha)')
pyplot.tight_layout()

pyplot.figure()
fex.scatter_plot(x_feat_key='NDVI', y_feat_key='AgcHa', class_key='DegrClass', xfn=lambda x: np.log10(x+1.))
pyplot.xlabel('NDVI')
pyplot.ylabel('AGC (tC/ha)')
pyplot.tight_layout()

vr.cleanup()
imr.cleanup()

X, y, feat_keys = fex.get_feat_array_ex(y_data_key='AgcHa')
Xselected_feats, selected_scores, selected_keys = mdl.FeatureSelector.forward_selection(X, y, feat_keys=feat_keys, max_num_feats=30, cv=5,
                                                                                        score_fn=lambda y,pred: -np.sqrt(metrics.mean_squared_error(y, pred)))

#------------------------------------------------------------------------------------------------------------------------
# make plots of num feats vs r2 / RMSE
r2 = np.zeros(selected_scores.__len__())
rmse = np.zeros(selected_scores.__len__())
rmse_ci = np.zeros((selected_scores.__len__(),2))
num_feats = np.arange(1, len(selected_scores)+1)
for i in range(0, selected_scores.__len__()):
    scores, predicted = mdl.FeatureSelector.score_model(Xselected_feats[:, :i + 1], y, model=linear_model.LinearRegression(), find_predicted=True, cv=X.shape[0])
    r2[i] = scores['R2_stacked']
    rmse_v = np.abs(scores['test_user'])/1000.
    rmse[i] = rmse_v.mean()
    rmse_ci[i,:] = np.percentile(rmse_v, [5, 95])
    print('.', end=' ')
print(' ')

# fontSize = 12.
# pyplot.rcParams.update({'font.size': fontSize})

# plots for report
fig = pyplot.figure()
fig.set_size_inches(8, 6, forward=True)
pyplot.subplot(2,1,1)
pyplot.plot(num_feats, r2, 'k-')
pyplot.xlabel('Number of features')
pyplot.ylabel('$\mathit{R}^2$')
# pyplot.grid()
pyplot.tight_layout()
pyplot.subplot(2,1,2)
pyplot.plot(num_feats, rmse, 'k-')
pyplot.xlabel('Number of features')
pyplot.ylabel('RMSE (t C ha$^{-1}$)')
# pyplot.grid()
pyplot.tight_layout()
fig.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\NgiAgcAccVsNumFeats1.png', dpi=300)

fig, ax1 = pyplot.subplots()
fig.set_size_inches(8, 4, forward=True)
color = 'tab:red'
ax1.set_xlabel('Number of features')
ax1.set_ylabel('$\mathit{R}^2$', color=color)  # we already handled the x-label with ax1
ax1.plot(num_feats, r2, color=color)
ax1.tick_params(axis='y', labelcolor=color)
# pyplot.grid()
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('-RMSE (t C ha$^{-1}$)', color=color)  # we already handled the x-label with ax1
ax2.plot(num_feats, -rmse, color=color)
ax2.tick_params(axis='y', labelcolor=color)
# pyplot.grid()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
pyplot.show()
fig.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\NgiAgcAccVsNumFeats2.png', dpi=300)

#------------------------------------------------------------------------------------------------------------------------
# report scatter plots for best and single feature models
print('\nBest model scores:')
best_model_idx = np.argmin(rmse)
scores, predicted = mdl.FeatureSelector.score_model(Xselected_feats[:, :best_model_idx + 1], old_div(y, 1000), model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=X.shape[0], print_scores=True)

print('\nBest model features:')
for k in selected_keys[:best_model_idx+1]:
    print(k)

# su.scatter_plot(y/1000., predicted/1000., labels=implot_feat_dict.keys())
fig = pyplot.figure()
fig.set_size_inches(5, 4, forward=True)
scatter_y_pred(y/1000., predicted, scores)
fig.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\MeasVsNgiPredAgcBestModel.png', dpi=300)

fig = pyplot.figure()
fig.set_size_inches(10, 4, forward=True)
pyplot.subplot(1, 2, 1)
scatter_y_pred(y/1000., predicted, scores)
pyplot.title('(a)')
pyplot.subplot(1, 2, 2)

print('\nBest single feature model scores:')
scores, predicted = mdl.FeatureSelector.score_model(Xselected_feats[:, :1], old_div(y, 1000), model=linear_model.LinearRegression(),
                                                    find_predicted=True, cv=X.shape[0], print_scores=True)
scatter_y_pred(y/1000., predicted, scores)
pyplot.title('(b)')
fig.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\MeasVsNgiPredAgcBestModels.png', dpi=300)
