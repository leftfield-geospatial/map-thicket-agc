"""
  GEF5-SLM: Above ground carbon estimation in thicket using multi-spectral images
  Copyright (C) 2020 Dugal Harris
  Released under GNU Affero General Public License (AGPL) (https://www.gnu.org/licenses/agpl.html)
  Email: dugalh@gmail.com
"""

import logging
import numpy as np
from sklearn import linear_model, metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EvaluateCalibration(object):
    def __init__(self, model_data_list=[], y=[], strata=None, calib_data_list=[], model=linear_model.LinearRegression):
        """
        Evaluate calibration by testing model performance over calibrated image data

        Parameters
        ----------
        model_data_list : list
            each element is a numpy.array_like containing model feature(s) from a single image
        y : numpy.array_like
            target / ground truth values for model
        strata : numpy.array_like
            strata for stratified bootstrap of calibration plots (optional)
        calib_data_list : list
            each element is a numpy.array_like containing calibration feature(s) from a single image
        model : sklearn.BaseEstimator
            model type to test
        """
        self.model_data_list = model_data_list
        self.calib_data_list = calib_data_list
        self.model = model
        self.y = y
        self.fitted_models = []
        self.strata = strata
        self.model_scores_array = None
        self.calib_scores_array = None

    def bootstrap(self, fit_model, fit_model_data, fit_calib_data, test_model_data, test_calib_data, n_bootstraps=10, n_calib_plots=10):
        """
        Repeated two-image calibration evaluations on random resamplings of image data.
        NOTE: Here, the image from which the model is derived is termed the "fit" image, and the new image to which model is
        applied is termed the "test" image.  Data used for modelling eg AGC in a single image is called "model" data.  Data
        used for fitting calibration transforms is termed "calib" data.  This terminology is used in naming the variables below.

        Parameters
        ----------
        fit_model : sklearn.BaseEstimator
            fitted model to test calibration of
        fit_model_data : numpy.array_like
            model image features from which fit_model is derived
        fit_calib_data : numpy.array_like
            calib image features to use for fitting calibration transform
        test_model_data : numpy.array_like
            model image features to use for testing model (should be independent of fit_model_data)
        test_calib_data : numpy.array_like
            calib image features to use for fitting calibration transform
        n_bootstraps : int
            number of times to bootstrap the fitting of the calibration transform
        n_calib_plots : int
            number of image plots (rows from fit_calib_data etc) to fit calibration transform to

        Returns
        -------
        model_scores : dict
            model accuracies over bootstraps {'r2':[...], 'rmse': [...]}
        calib_scores : dict
            calibration accuracies over bootstraps {'r2':[...], 'rmse': [...]}
        """

        r2_model = np.zeros((n_bootstraps, 1))
        rmse_model = np.zeros((n_bootstraps, 1))

        r2_calib = np.zeros((n_bootstraps, test_calib_data.shape[1]))
        rmse_calib = np.zeros((n_bootstraps, test_calib_data.shape[1]))

        # sample with bootstrapping the calib plots, fit calib models, apply to model_data and test
        for bi in range(0, n_bootstraps):
            if self.strata is None:
                calib_plot_idx = np.random.permutation(len(test_model_data))[:n_calib_plots]
            else:   # stratified random sampling balanced over strata
                calib_plot_idx = []
                strata_list = np.unique(self.strata)
                for strata_i in strata_list:
                    strata_idx = np.int32(np.where(strata_i == self.strata)[0])
                    calib_plot_idx += np.random.permutation(strata_idx)[
                                      :np.round(n_calib_plots / len(strata_list)).astype('int')].tolist()

                calib_plot_idx = np.array(calib_plot_idx)

            # test_plot_idx = np.setdiff1d(np.arange(0, len(self.y)), calib_plot_idx)   # exclude the calib plots
            test_plot_idx = np.arange(0, len(self.y))   # include the calib plots
            calib_feats = np.zeros((len(test_plot_idx), test_calib_data.shape[1]))
            for fi in range(0, test_calib_data.shape[1]):   # loop through features and calibrate each one
                calib_model = linear_model.LinearRegression()   # fit calibration transform between test and fit images
                calib_model.fit(test_calib_data[calib_plot_idx, fi].reshape(-1, 1),
                                fit_calib_data[calib_plot_idx, fi].reshape(-1, 1))

                calib_feat = calib_model.predict(test_model_data[test_plot_idx, fi].reshape(-1, 1))     # calibrate test feature
                r2_calib[bi, fi] = metrics.r2_score(fit_model_data[test_plot_idx, fi], calib_feat)      # find R2 between fit and calibrated test feature
                rmse_calib[bi, fi] = np.sqrt(metrics.mean_squared_error(fit_model_data[test_plot_idx, fi], calib_feat)) # find RMSE between fit and calibrated test feature
                calib_feats[:, fi] = calib_feat.flatten()   # store calibrated feature for model eval below

            predicted = fit_model.predict(calib_feats)      # find model prediction on calibrated test features
            r2_model[bi] = metrics.r2_score(self.y[test_plot_idx], predicted)   # find R2 between model predictions and ground truth
            rmse_model[bi] = np.sqrt(metrics.mean_squared_error(self.y[test_plot_idx], predicted))  # find RMSE on model predictions  and ground truth

        model_scores = {'r2':r2_model, 'rmse': rmse_model}
        calib_scores = {'r2':r2_calib, 'rmse': rmse_calib}
        return model_scores, calib_scores

    def test(self, n_bootstraps=10, n_calib_plots=10):
        """
        Bootstrapped testing of model and calibration accuracies over provided images

        Parameters
        ----------
        n_bootstraps : int
            number of bootstraps (random resamplings) to test over
        n_calib_plots : int
            number of plots to use for fitting calibration transform

        Returns
        -------
        model_scores_array: numpy.array_like
            model_scores_array[i,j] is the score of the model derived from the ith image and calibrated to the jth image
        model_scores_array: numpy.array_like
            calib_scores_array[i,j] is the score of the calibration transform from the ith image to the jth image
        """
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
                logger.info('mean(R^2) : {0:.4f}'.format(model_scores['r2'].mean()))
                logger.info('std(R^2)  : {0:.4f}'.format(model_scores['r2'].std()))
                logger.info('mean(RMSE): {0:.4f}'.format(model_scores['rmse'].mean()))
                logger.info('std(RMSE) : {0:.4f}'.format(model_scores['rmse'].std()))
                logger.info('Calib scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                logger.info('mean(R^2) : {0:.4f}'.format(calib_scores['r2'].mean(axis=0)))
                logger.info('std(R^2)  : {0:.4f}'.format(calib_scores['r2'].std(axis=0)))
                logger.info('mean(RMSE): {0:.4f}'.format(calib_scores['rmse'].mean(axis=0)))
                logger.info('std(RMSE) : {0:.4f}'.format(calib_scores['rmse'].std(axis=0)))
        return self.model_scores_array, self.calib_scores_array

    def print_scores(self):
        for scores_array, label in zip([self.model_scores_array, self.calib_scores_array], ['Model', 'Calib']):
            for key in ['mean(r2)', 'std(r2)', 'mean(rmse)', 'std(rmse)']:
                print('{0} {1}:'.format(label, key))
                score_array = np.zeros(scores_array.shape)
                for ri in range(scores_array.shape[0]):
                    for ci in range(scores_array.shape[1]):
                        if scores_array[ri, ci] is None or type(scores_array[ri, ci]) is int:
                            score_array[ri, ci] = 0.
                        else:
                            score_array[ri, ci] = scores_array[ri, ci][key]
                print(score_array)
                overall_mean = np.diag(np.flipud(score_array)).mean()
                print('Overall mean({0} {1}): {2:0.4f}'.format(label, key, overall_mean))
                print(' ')

