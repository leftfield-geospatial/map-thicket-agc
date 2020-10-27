from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div

import warnings, logging
import numpy as np
import rasterio
from sklearn import linear_model, metrics
from modules import modelling as mdl

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ModelCalibrationTest(object):
    # def __init__(self, plot_featdict_list=[], y_key='', calib_featdict_list=[], feat_keys='', model_feat_keys=['r_n'], model=linear_model.LinearRegression()):
    #     self.plot_data_list = plot_data_list
    #     self.calib_data_list = calib_data_list
    #     self.feat_keys = feat_keys
    #     self.model_feat_idx = model_feat_idx
    #     self.model = model
    #     self.y = y
    #     self.fitted_models = []

    def __init__(self, plot_data_list=[], y=[], strata=None, calib_data_list=[], feat_keys='', model_feat_idx=[0], model=linear_model.LinearRegression):
        self.plot_data_list = plot_data_list
        self.calib_data_list = calib_data_list
        self.feat_keys = feat_keys
        self.model_feat_idx = model_feat_idx
        self.model = model
        self.y = y
        self.fitted_models = []
        self.strata = strata
        self.model_scores_array = None
        self.calib_scores_array = None

    def BootStrapCalibration(self, fit_model, fit_plot_data, fit_calib_data, test_plot_data, test_calib_data,
                                     n_bootstraps=10, n_calib_plots=10):
        r2_model = np.zeros((n_bootstraps, 1))
        rmse_model = np.zeros((n_bootstraps, 1))

        r2_calib = np.zeros((n_bootstraps, self.model_feat_idx.__len__()))
        rmse_calib = np.zeros((n_bootstraps, self.model_feat_idx.__len__()))
        # TO DO: make a sub-function
        # sample with bootstrapping the calib plots, fit calib models, apply to plot_data and test
        for bi in range(0, n_bootstraps):
            if self.strata is None:
                calib_plot_idx = np.random.permutation(test_plot_data.__len__())[:n_calib_plots]
            else:
                calib_plot_idx = []
                strata_list = np.unique(self.strata)
                for strata_i in strata_list:
                    strata_idx = np.int32(np.where(strata_i == self.strata)[0])
                    calib_plot_idx += np.random.permutation(strata_idx)[
                                      :np.round(old_div(n_calib_plots, strata_list.__len__()))].tolist()

                calib_plot_idx = np.array(calib_plot_idx)
            calib_feats = []
            # loop through features in model_feat_idx and calibrate each one
            calib_feats = np.zeros((test_calib_data.shape[0], self.model_feat_idx.__len__()))
            for fi, feat_idx in enumerate(self.model_feat_idx):
                calib_model = linear_model.LinearRegression()
                calib_model.fit(test_calib_data[calib_plot_idx, feat_idx].reshape(-1, 1),
                                fit_calib_data[calib_plot_idx, feat_idx].reshape(-1, 1))
                calib_feat = calib_model.predict(test_plot_data[:, feat_idx].reshape(-1, 1))
                r2_calib[bi, fi] = metrics.r2_score(fit_plot_data[:, feat_idx], calib_feat)
                rmse_calib[bi, fi] = np.sqrt(
                    metrics.mean_squared_error(fit_plot_data[:, feat_idx], calib_feat))
                calib_feats[:, fi] = calib_feat.flatten()

            # calib_feats = np.array(calib_feats).transpose()
            predicted = fit_model.predict(calib_feats)
            r2_model[bi] = metrics.r2_score(self.y, predicted)
            rmse_model[bi] = np.sqrt(metrics.mean_squared_error(self.y, predicted))

        model_scores = {'r2':r2_model, 'rmse': rmse_model}
        calib_scores = {'r2':r2_calib, 'rmse': rmse_calib}
        return model_scores, calib_scores

    def TestCalibration(self, n_bootstraps=10, n_calib_plots=10):
        np.set_printoptions(precision=4)
        self.fitted_models = []
        for plot_data in self.plot_data_list:
            fit_model = self.model()
            fit_model.fit(plot_data[:, self.model_feat_idx], self.y)
            self.fitted_models.append(fit_model)

        model_idx = list(range(0, self.fitted_models.__len__()))
        # loop over different images/models to fit to
        self.model_scores_array = np.zeros((self.fitted_models.__len__(), self.fitted_models.__len__()), dtype=object)
        self.calib_scores_array = np.zeros((self.fitted_models.__len__(), self.fitted_models.__len__()), dtype=object)

        for fmi in model_idx:
            fit_model = self.fitted_models[fmi]
            test_model_idx = np.setdiff1d(model_idx, fmi)
            fit_calib_data = self.calib_data_list[fmi]
            fit_plot_data = self.plot_data_list[fmi]
            # loop over the remaining images/models to test calibration on
            for tmi in test_model_idx:
                test_calib_data = self.calib_data_list[tmi]
                test_plot_data = self.plot_data_list[tmi]

                model_scores, calib_scores = self.BootStrapCalibration(fit_model, fit_plot_data, fit_calib_data, test_plot_data,
                                                           test_calib_data, n_bootstraps=n_bootstraps, n_calib_plots=n_calib_plots)
                self.model_scores_array[fmi, tmi] = {'mean(r2)': model_scores['r2'].mean(), 'std(r2)': model_scores['r2'].std(),
                                                'mean(rmse)': model_scores['rmse'].mean(), 'std(rmse)': model_scores['rmse'].std()}
                self.calib_scores_array[fmi, tmi] = {'mean(r2)': calib_scores['r2'].mean(axis=0), 'std(r2)': calib_scores['r2'].std(axis=0),
                                                'mean(rmse)': calib_scores['rmse'].mean(axis=0), 'std(rmse)': calib_scores['rmse'].std(axis=0)}
                logger.info('Model scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                logger.info('mean(R^2): {0:.4f}'.format(model_scores['r2'].mean()))
                logger.info('std(R^2): {0:.4f}'.format(model_scores['r2'].std()))
                logger.info('mean(RMSE): {0:.4f}'.format(model_scores['rmse'].mean()))
                logger.info('std(RMSE): {0:.4f}'.format(model_scores['rmse'].std()))
                logger.info(' ')
                logger.info('Calib scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                logger.info('mean(R^2): {0}'.format(calib_scores['r2'].mean(axis=0)))
                logger.info('std(R^2): {0}'.format(calib_scores['r2'].std(axis=0)))
                logger.info('mean(RMSE): {0}'.format(calib_scores['rmse'].mean(axis=0)))
                logger.info('std(RMSE): {0}'.format(calib_scores['rmse'].std(axis=0)))
                logger.info(' ')
        return self.model_scores_array, self.calib_scores_array

    def PrintScores(self):
        for scores_array, label in zip([self.model_scores_array, self.calib_scores_array], ['Model', 'Calib']):
            for key in ['mean(r2)', 'std(r2)', 'mean(rmse)', 'std(rmse)']:
                logger.info('{0} {1}:'.format(label, key))
                score_array = np.zeros(scores_array.shape)
                for ri in range(scores_array.shape[0]):
                    for ci in range(scores_array.shape[1]):
                        if scores_array[ri, ci] is None:
                            score_array[ri, ci] = 0.
                        else:
                            score_array[ri, ci] = scores_array[ri, ci][key]
                logger.info(score_array)
                overall_mean = np.diag(np.flipud(score_array)).mean()
                logger.info('Overall mean({0} {1}): {2:0.4f}'.format(label, key, overall_mean))
                logger.info(' ')

        return


class ModelCalibrationTestEx(object):
    # def __init__(self, plot_featdict_list=[], y_key='', calib_featdict_list=[], feat_keys='', model_feat_keys=['r_n'], model=linear_model.LinearRegression()):
    #     self.plot_data_list = plot_data_list
    #     self.calib_data_list = calib_data_list
    #     self.feat_keys = feat_keys
    #     self.model_feat_idx = model_feat_idx
    #     self.model = model
    #     self.y = y
    #     self.fitted_models = []

    def __init__(self, plot_data_list=[], y=[], strata=None, calib_data_list=[], model=linear_model.LinearRegression):
        self.plot_data_list = plot_data_list
        self.calib_data_list = calib_data_list
        self.model = model
        self.y = y
        self.fitted_models = []
        self.strata = strata
        self.model_scores_array = None
        self.calib_scores_array = None

    def BootStrapCalibration(self, fit_model, fit_plot_data, fit_calib_data, test_plot_data, test_calib_data,
                                     n_bootstraps=10, n_calib_plots=10):
        r2_model = np.zeros((n_bootstraps, 1))
        rmse_model = np.zeros((n_bootstraps, 1))

        r2_calib = np.zeros((n_bootstraps, test_calib_data.shape[1]))
        rmse_calib = np.zeros((n_bootstraps, test_calib_data.shape[1]))
        # TO DO: make a sub-function
        # sample with bootstrapping the calib plots, fit calib models, apply to plot_data and test
        for bi in range(0, n_bootstraps):
            if self.strata is None:
                calib_plot_idx = np.random.permutation(test_plot_data.__len__())[:n_calib_plots]
            else:
                calib_plot_idx = []
                strata_list = np.unique(self.strata)
                for strata_i in strata_list:
                    strata_idx = np.int32(np.where(strata_i == self.strata)[0])
                    calib_plot_idx += np.random.permutation(strata_idx)[
                                      :np.round(old_div(n_calib_plots, strata_list.__len__()))].tolist()

                calib_plot_idx = np.array(calib_plot_idx)
            # test_plot_idx = np.setdiff1d(np.arange(0, len(self.y)), calib_plot_idx)   # exclude the calib plots
            test_plot_idx = np.arange(0, len(self.y))   # include the calib plots
            calib_feats = []
            # loop through features in model_feat_idx and calibrate each one
            # calib_feats = np.zeros((test_calib_data.shape[0], test_calib_data.shape[1]))
            calib_feats = np.zeros((len(test_plot_idx), test_calib_data.shape[1]))
            for fi in range(0, test_calib_data.shape[1]):
                calib_model = linear_model.LinearRegression()
                calib_model.fit(test_calib_data[calib_plot_idx, fi].reshape(-1, 1),
                                fit_calib_data[calib_plot_idx, fi].reshape(-1, 1))

                calib_feat = calib_model.predict(test_plot_data[test_plot_idx, fi].reshape(-1, 1))
                r2_calib[bi, fi] = metrics.r2_score(fit_plot_data[test_plot_idx, fi], calib_feat)
                rmse_calib[bi, fi] = np.sqrt(
                    metrics.mean_squared_error(fit_plot_data[test_plot_idx, fi], calib_feat))
                calib_feats[:, fi] = calib_feat.flatten()

            # calib_feats = np.array(calib_feats).transpose()
            predicted = fit_model.predict(calib_feats)
            r2_model[bi] = metrics.r2_score(self.y[test_plot_idx], predicted)
            rmse_model[bi] = np.sqrt(metrics.mean_squared_error(self.y[test_plot_idx], predicted))

        model_scores = {'r2':r2_model, 'rmse': rmse_model}
        calib_scores = {'r2':r2_calib, 'rmse': rmse_calib}
        return model_scores, calib_scores

    def TestCalibration(self, n_bootstraps=10, n_calib_plots=10):
        np.set_printoptions(precision=4)
        self.fitted_models = []
        for plot_data in self.plot_data_list:
            fit_model = self.model()
            fit_model.fit(plot_data, self.y)
            self.fitted_models.append(fit_model)

        model_idx = list(range(0, self.fitted_models.__len__()))
        # loop over different images/models to fit to
        self.model_scores_array = np.zeros((self.fitted_models.__len__(), self.fitted_models.__len__()), dtype=object)
        self.calib_scores_array = np.zeros((self.fitted_models.__len__(), self.fitted_models.__len__()), dtype=object)

        for fmi in model_idx:
            fit_model = self.fitted_models[fmi]
            test_model_idx = np.setdiff1d(model_idx, fmi)
            fit_calib_data = self.calib_data_list[fmi]
            fit_plot_data = self.plot_data_list[fmi]
            # loop over the remaining images/models to test calibration on
            for tmi in test_model_idx:
                test_calib_data = self.calib_data_list[tmi]
                test_plot_data = self.plot_data_list[tmi]

                model_scores, calib_scores = self.BootStrapCalibration(fit_model, fit_plot_data, fit_calib_data, test_plot_data,
                                                           test_calib_data, n_bootstraps=n_bootstraps, n_calib_plots=n_calib_plots)
                self.model_scores_array[fmi, tmi] = {'mean(r2)': model_scores['r2'].mean(), 'std(r2)': model_scores['r2'].std(),
                                                'mean(rmse)': model_scores['rmse'].mean(), 'std(rmse)': model_scores['rmse'].std()}
                self.calib_scores_array[fmi, tmi] = {'mean(r2)': calib_scores['r2'].mean(axis=0), 'std(r2)': calib_scores['r2'].std(axis=0),
                                                'mean(rmse)': calib_scores['rmse'].mean(axis=0), 'std(rmse)': calib_scores['rmse'].std(axis=0)}
                logger.info('Model scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                logger.info('mean(R^2): {0:.4f}'.format(model_scores['r2'].mean()))
                logger.info('std(R^2): {0:.4f}'.format(model_scores['r2'].std()))
                logger.info('mean(RMSE): {0:.4f}'.format(model_scores['rmse'].mean()))
                logger.info('std(RMSE): {0:.4f}'.format(model_scores['rmse'].std()))
                logger.info(' ')
                logger.info('Calib scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                logger.info('mean(R^2): {0}'.format(calib_scores['r2'].mean(axis=0)))
                logger.info('std(R^2): {0}'.format(calib_scores['r2'].std(axis=0)))
                logger.info('mean(RMSE): {0}'.format(calib_scores['rmse'].mean(axis=0)))
                logger.info('std(RMSE): {0}'.format(calib_scores['rmse'].std(axis=0)))
                logger.info(' ')
        return self.model_scores_array, self.calib_scores_array

    def PrintScores(self):
        for scores_array, label in zip([self.model_scores_array, self.calib_scores_array], ['Model', 'Calib']):
            for key in ['mean(r2)', 'std(r2)', 'mean(rmse)', 'std(rmse)']:
                logger.info('{0} {1}:'.format(label, key))
                score_array = np.zeros(scores_array.shape)
                for ri in range(scores_array.shape[0]):
                    for ci in range(scores_array.shape[1]):
                        if scores_array[ri, ci] is None or type(scores_array[ri, ci]) is int:
                            score_array[ri, ci] = 0.
                        else:
                            score_array[ri, ci] = scores_array[ri, ci][key]
                logger.info(score_array)
                overall_mean = np.diag(np.flipud(score_array)).mean()
                logger.info('Overall mean({0} {1}): {2:0.4f}'.format(label, key, overall_mean))
                logger.info(' ')

        return

