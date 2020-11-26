"""
  GEF5-SLM: Above ground carbon estimation in thicket using multi-spectral images
  Copyright (C) 2020 Dugal Harris
  Released under GNU Affero General Public License (AGPL) (https://www.gnu.org/licenses/agpl.html)
  Email: dugalh@gmail.com
"""

import logging
import numpy as np
from sklearn import linear_model, metrics
from agc_estimation import imaging as img

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# params: calib_plots from >1 data set
#       model_data_plots from >1 data set
#       (for now the above are the same thing)
#       1 data_set is specified as fitted one, the rest are tests, this can also be done sort of cross-validated
#       a model spec i.e. feature indices and model type
#       num calib plots to use


class EvaluateCalibration(object):
    # def __init__(self, plot_featdict_list=[], y_key='', calib_featdict_list=[], feat_keys='', model_feat_keys=['r_n'], model=linear_model.LinearRegression()):
    #     self.model_data_list = model_data_list
    #     self.calib_data_list = calib_data_list
    #     self.feat_keys = feat_keys
    #     self.model_feat_idx = model_feat_idx
    #     self.model = model
    #     self.y = y
    #     self.fitted_models = []

    def __init__(self, model_data_list=[], y=[], strata=None, calib_data_list=[], model=linear_model.LinearRegression):
        """
        Evaluate calibration by testing model performance in calibrated images
        Parameters
        ----------
        model_data_list : list
            list of numpy.array_like's containing model feature(s)
        y : numpy.array_like
            model target / dependent variables
        strata : numpy.array_like
            strata for stratified bootstrap of calibration plots (optional)
        calib_data_list : list
            list of numpy.array_like's containing calibration feature(s)
        model : sklearn.BaseEstimator
            model type to test with
        """
        self.model_data_list = model_data_list
        self.calib_data_list = calib_data_list
        self.model = model
        self.y = y
        self.fitted_models = []
        self.strata = strata
        self.model_scores_array = None
        self.calib_scores_array = None

    def bootstrap(self, fit_model, fit_model_data, fit_calib_data, test_model_data, test_calib_data,
                  n_bootstraps=10, n_calib_plots=10):
        r2_model = np.zeros((n_bootstraps, 1))
        rmse_model = np.zeros((n_bootstraps, 1))

        r2_calib = np.zeros((n_bootstraps, test_calib_data.shape[1]))
        rmse_calib = np.zeros((n_bootstraps, test_calib_data.shape[1]))
        # TO DO: make a sub-function
        # sample with bootstrapping the calib plots, fit calib models, apply to model_data and test
        for bi in range(0, n_bootstraps):
            if self.strata is None:
                calib_plot_idx = np.random.permutation(len(test_model_data))[:n_calib_plots]
            else:
                calib_plot_idx = []
                strata_list = np.unique(self.strata)    # TODO: groupby
                for strata_i in strata_list:
                    strata_idx = np.int32(np.where(strata_i == self.strata)[0])
                    calib_plot_idx += np.random.permutation(strata_idx)[
                                      :np.round(n_calib_plots / len(strata_list)).astype('int')].tolist()

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

                calib_feat = calib_model.predict(test_model_data[test_plot_idx, fi].reshape(-1, 1))
                r2_calib[bi, fi] = metrics.r2_score(fit_model_data[test_plot_idx, fi], calib_feat)
                rmse_calib[bi, fi] = np.sqrt(
                    metrics.mean_squared_error(fit_model_data[test_plot_idx, fi], calib_feat))
                calib_feats[:, fi] = calib_feat.flatten()

            # calib_feats = np.array(calib_feats).transpose()
            predicted = fit_model.predict(calib_feats)
            r2_model[bi] = metrics.r2_score(self.y[test_plot_idx], predicted)
            rmse_model[bi] = np.sqrt(metrics.mean_squared_error(self.y[test_plot_idx], predicted))

        model_scores = {'r2':r2_model, 'rmse': rmse_model}
        calib_scores = {'r2':r2_calib, 'rmse': rmse_calib}
        return model_scores, calib_scores

    def test(self, n_bootstraps=10, n_calib_plots=10):
        np.set_printoptions(precision=4)
        self.fitted_models = []
        for model_data in self.model_data_list:
            fit_model = self.model()
            fit_model.fit(model_data, self.y)
            self.fitted_models.append(fit_model)

        model_idx = list(range(0, self.fitted_models.__len__()))
        # loop over different images/models to fit to
        nm = len(self.fitted_models)
        self.model_scores_array = np.zeros((nm, nm), dtype=object)
        self.calib_scores_array = np.zeros((nm, nm), dtype=object)

        for fmi in model_idx:
            fit_model = self.fitted_models[fmi]
            test_model_idx = np.setdiff1d(model_idx, fmi)
            fit_calib_data = self.calib_data_list[fmi]
            fit_model_data = self.model_data_list[fmi]
            # loop over the remaining images/models to test calibration on
            for tmi in test_model_idx:
                test_calib_data = self.calib_data_list[tmi]
                test_model_data = self.model_data_list[tmi]

                model_scores, calib_scores = self.bootstrap(fit_model, np.array(fit_model_data), np.array(fit_calib_data), np.array(test_model_data),
                                                            np.array(test_calib_data), n_bootstraps=n_bootstraps, n_calib_plots=n_calib_plots)
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

    def print_scores(self):
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

