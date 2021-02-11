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


from map_thicket_agc import get_logger
import numpy as np
from sklearn import linear_model, metrics
import pandas as pd

logger = get_logger(__name__)

class EvaluateCalibration(object):
    def __init__(self, model_data_dict: dict=None, y=None, calib_data_dict: dict=None, calib_strata=None,
                 model=linear_model.LinearRegression):
        """
        Evaluate calibration by testing model performance over calibrated image data

        Parameters
        ----------
        model_data_dict : dict
            each value is a numpy.array_like containing model feature(s) from a single image, keys are labels for images
        y : numpy.array_like
            target / ground truth values for model
        calib_strata : numpy.array_like
            strata for stratified bootstrap of calibration plots (optional)
        calib_data_dict : list
            each value element is a numpy.array_like containing calibration feature(s) from a single image, keys are labels for images
        model : sklearn.BaseEstimator
            model type to test
        """
        self.model_data_dict = model_data_dict
        self.calib_data_dict = calib_data_dict
        self.model = model
        self.y = y
        self.calib_strata = calib_strata
        self.model_scores_df = None
        self.calib_scores_df = None
        self._fitted_models_dict = {}

    def _bootstrap(self, fit_model, fit_model_data, fit_calib_data, test_model_data, test_calib_data, n_bootstraps=10, n_calib_plots=10):
        """
        Repeated two-image calibration evaluations on random resamplings of image data.
        NOTE: Here, the image from which the model is derived is termed the "fit" image, and the new image to which model
        is applied is termed the "test" image.  Data used for modelling eg AGC in a single image is called "model" data.
        Data used for fitting calibration transforms is termed "calib" data.  This terminology is used in naming the
        variables below.

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
            if self.calib_strata is None:
                calib_plot_idx = np.random.permutation(len(test_calib_data))[:n_calib_plots]
            else:   # stratified random sampling balanced over strata
                calib_plot_idx = []
                strata_list = np.unique(self.calib_strata)
                for strata_i in strata_list:
                    strata_idx = np.int32(np.where(strata_i == self.calib_strata)[0])
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
                # print(f'slope: {calib_model.coef_}, intercept: {calib_model.intercept_}')
            predicted = fit_model.predict(calib_feats)      # find model prediction on calibrated test features
            r2_model[bi] = metrics.r2_score(self.y[test_plot_idx], predicted)   # find R2 between model predictions and ground truth
            rmse_model[bi] = np.sqrt(metrics.mean_squared_error(self.y[test_plot_idx], predicted))  # find RMSE on model predictions  and ground truth

        model_scores = {'r2':r2_model.mean(), 'rmse': rmse_model.mean(), 'std(r2)':r2_model.std(), 'std(rmse)': rmse_model.std()}
        calib_scores = {'r2':r2_calib.mean(), 'rmse': rmse_calib.mean(), 'std(r2)':r2_calib.std(), 'std(rmse)': rmse_calib.std()}
        # calib_scores = {'r2':r2_calib, 'rmse': rmse_calib}
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
        self._fitted_models_dict = {}
        for fit_model_key, model_data in self.model_data_dict.items():
            fit_model = self.model()
            fit_model.fit(model_data, self.y)
            self._fitted_models_dict[fit_model_key] = fit_model

        model_keys = list(self._fitted_models_dict.keys())

        score_keys = ['r2', 'rmse', 'std(r2)', 'std(rmse)']
        multi_index = pd.MultiIndex.from_product([score_keys, model_keys])
        self.model_scores_df = pd.DataFrame(columns=model_keys, index=multi_index)
        self.calib_scores_df = pd.DataFrame(columns=model_keys, index=multi_index)
        for fit_model_key in model_keys:    # loop over images/models to fit to
            fit_model = self._fitted_models_dict[fit_model_key]
            test_model_keys = np.setdiff1d(model_keys, fit_model_key)
            fit_calib_data = self.calib_data_dict[fit_model_key]
            fit_model_data = self.model_data_dict[fit_model_key]
            # loop over the remaining images/models to test calibration on
            for test_model_key in test_model_keys:
                test_calib_data = self.calib_data_dict[test_model_key]
                test_model_data = self.model_data_dict[test_model_key]

                model_scores, calib_scores = self._bootstrap(fit_model, np.array(fit_model_data), np.array(fit_calib_data), np.array(test_model_data),
                                                             np.array(test_calib_data), n_bootstraps=n_bootstraps, n_calib_plots=n_calib_plots)
                for k in model_scores.keys():   # to get around pandas weird view vs copy and multiindex stuff
                    self.model_scores_df.loc[(k, fit_model_key), test_model_key] = model_scores[k]
                    self.calib_scores_df.loc[(k, fit_model_key), test_model_key] = calib_scores[k]
        self.print_scores()
        return self.model_scores_df, self.calib_scores_df

    def print_scores(self):
        logger.info('print_scores')
        with pd.option_context('display.float_format', '{:0.4f}'.format):
            for scores_label, scores_df in zip(['Model', 'Calib'], [self.model_scores_df, self.calib_scores_df]):
                for scores_key in ['r2', 'std(r2)', 'rmse', 'std(rmse)']:
                    logger.info(' ')
                    logger.info(f'{scores_label} - {scores_key}:')
                    logger.info('\n' + scores_df.loc[scores_key].to_string())
