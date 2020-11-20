#
   """
      GEF5-SLM Above ground carbon estimation in thicket using multi-spectral images
      Copyright (C) 2020 Dugal Harris
      Released under GNU Affero General Public License (AGPL) (https://www.gnu.org/licenses/agpl.html)
      email dugalh@gmail.com
   """




#

      GEF5-SLM Above ground carbon estimation in thicket using multi-spectral images
      Copyright (C) 2020  Dugal Harris

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU Affero General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU Affero General Public License for more details.

      You should have received a copy of the GNU Affero General Public License
      along with this program.  If not, see <https://www.gnu.org/licenses/>.



from __future__ import print_function
from __future__ import division

#################################################################################################################
# test ModelCalibrationTest for all images in array notation
from past.utils import old_div
from agc_estimation import imaging as su
import numpy as np
from sklearn import linear_model

# reload(su)

samplingPlotGtFile = r"C:\Data\Development\Projects\GEF-5 SLM\Data\Outputs\Geospatial\GEF Plot Polygons with AGC.shp"

# new py 3 May 2020 files
imageFiles = [r"D:\OneDrive\GEF Essentials\Source Images\WorldView3 Oct 2017\WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif",
              r"D:\OneDrive\GEF Essentials\Source Images\WorldView3 Nov 2018\WorldView3_Nov2018_OrthoThinSpline_NoAtcor_PanSharpMs.tif",
              r"D:\OneDrive\GEF Essentials\Source Images\WorldView3 Aug 2017\WorldView3_Aug2017_OrthoThinSpline_NoAtcor_PanSharpMs.tif",
              r"D:\OneDrive\GEF Essentials\Source Images\NGI April 2015\Ngi_May2015_OrthoNgiDem_Xcalib_Rgbn_Mosaic.vrt"]

vr = su.GdalVectorReader(samplingPlotGtFile)
ld = vr.read()

image_readers = [su.GdalImageReader(imageFile) for imageFile in imageFiles]
feature_extractors = [su.MsImageFeatureExtractor(image_reader=imr, plot_feat_dict=ld['GEF Plot Polygons with AGC']) for imr in image_readers]
implot_feat_dicts = [fex.extract_image_features(patch_fn=su.MsImageFeatureExtractor.extract_patch_ms_features_ex) for fex in feature_extractors]
vr.cleanup()
for imr in image_readers:
    imr.cleanup()

# find the best single features for Wv3 2017 AGC modelling

# check correlation of features between 2018 WV3 and other images to see which are best

# set DegrClass field in implot_feat_dict using plot ID
for feat_dict in implot_feat_dicts:
    for f in list(feat_dict.values()):
        id = f['ID']
        if id[0] == 'S' or id[:3] == 'TCH':
            f['DegrClass'] = 'Severe'
        elif id[0] == 'M':
            f['DegrClass'] = 'Moderate'
        elif id[0] == 'P' or id[:3] == 'INT':
            f['DegrClass'] = 'Pristine'
        else:
            f['DegrClass'] = '?'

if False:   # find best single feat models so that we know which feats to try calibrate with
    X, y, feat_keys = feature_extractors[0].get_feat_array_ex(y_data_key='AgcHa')
    r2 = []
    rmse = []
    from collections import OrderedDict
    single_feat_model_scores = OrderedDict()
    for ki, key in enumerate(feat_keys):
        xv = X[:, ki]
        scores, predicted = su.FeatureSelector.score_model( X[:, ki].reshape(-1, 1), y / 1000.,
                                                           model=linear_model.LinearRegression(),
                                                           find_predicted=True, cv=10, print_scores=False)
        single_feat_model_scores[key] = {'r2': scores['R2_stacked'], 'rmse': -scores['test_user'].mean()}
        print('.', end=' ')

    r2 = np.array([sfms['r2'] for sfms in list(single_feat_model_scores.values())])
    rmse = np.array([sfms['rmse'] for sfms in list(single_feat_model_scores.values())])
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
        feat_scores[key] = OrderedDict({'WV3 2017 AGC R2':single_feat_model_scores[key]['r2']})

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

X_list = []
calib_feat_keys = ['Log(R/pan)']
for fex in feature_extractors:
    X, y, feat_keys = fex.get_feat_array_ex(y_data_key='AgcHa')
    feat_idx = []
    for calib_feat_key in calib_feat_keys:
        feat_idx.append(np.argwhere(feat_keys == calib_feat_key)[0][0])
    X_list.append(X[:, feat_idx])

classes = np.array([plot['DegrClass'] for plot in list(implot_feat_dicts[0].values())])

    # feat_idx = [13, 19]
# reload(su)
mct = su.ModelCalibrationTestEx(plot_data_list=X_list, y=old_div(y,1000), strata=classes, calib_data_list=X_list, model=linear_model.LinearRegression)

model_scores, calib_scores = mct.TestCalibration(n_bootstraps=100, n_calib_plots=9)
mct.PrintScores()
