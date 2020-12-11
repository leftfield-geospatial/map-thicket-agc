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
import sys
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict, cross_validate
import geopandas as gpd
import pandas as pd

if sys.version_info.major == 3:
    from sklearn.metrics import make_scorer
else:
    from sklearn.metrics.scorer import make_scorer

logger = get_logger(__name__)

def forward_selection(feat_df, y, max_num_feats=0, model=linear_model.LinearRegression(),
                      score_fn=None, cv=None):
    """
    Forward selection of features from a pandas dataframe, using cross-validation

    Parameters
    ----------
    feat_df : pandas.DataFrame
        features data only
    y : numpy.array_like
        target/output values corresponding to feat_df
    max_num_feats : int
        maximum number of features to select (default = select all)
    model : sklearn.BaseEstimator
        model for feature evaluation (default = LinearRegression)
    score_fn : function in form score = score_fn(y_true, y_pred)
        a model score function in the form of a sklearn metric (eg RMSE) to evaluate model (default = -RMSE)
    cv : int
        number of cross-validated folds to use (default = )

    Returns
    -------
    (selected_feats_df, selected_scores)
    selected_feats_df : pandas.DataFrame
        selected features
    selected_scores : list
        list of score dicts
    """

    if max_num_feats == 0:
        max_num_feats = feat_df.shape[1]
    selected_feats_df = gpd.GeoDataFrame(index=feat_df.index)   # remember order items are added
    selected_scores = []
    available_feats_df = feat_df.copy()

    logger.info('Forward selection: ')
    if score_fn is None:
        logger.info('Using negative RMSE score')
    else:
        logger.info('Using user score')
    while selected_feats_df.shape[1] < max_num_feats:
        best_score = -np.inf
        best_feat = []
        best_key = available_feats_df.columns[0]
        for feat_key, feat_vec in available_feats_df.iteritems():
            test_feats_df = pd.concat((selected_feats_df, feat_vec), axis=1, ignore_index=False) # list(selected_feats.values()) + [feat_vec]
            scores, predicted = score_model(test_feats_df, y, model=model,
                                                            score_fn=score_fn, cv=cv, find_predicted=False)
            if score_fn is None:
                score = -np.sqrt((scores['test_-RMSE']**2).mean())       # NB not mean(sqrt(RMSE))
            else:
                score = scores['test_user'].mean()

            if score > best_score:
                best_score = score
                best_feat = list(feat_vec)
                best_key = feat_key
        selected_feats_df[best_key] = best_feat
        selected_scores.append(best_score)
        available_feats_df.pop(best_key)
        logger.info('Feature {0} of {1}: {2}, Score: {3:.1f}'.format(selected_feats_df.shape[1], max_num_feats, best_key, best_score))
    # logger.info(' ')
    selected_scores = np.array(selected_scores)
    selected_feat_keys = selected_feats_df.columns
    best_selected_feat_keys = selected_feat_keys[:np.argmax(selected_scores) + 1]
    logger.info('Best score: {0}'.format(selected_scores.max()))
    logger.info('Num feats at best score: {0}'.format(np.argmax(selected_scores) + 1))
    logger.info('Feat keys at best score: {0}'.format(best_selected_feat_keys))

    return selected_feats_df, selected_scores

def ranking(feat_df, y, model=linear_model.LinearRegression(), score_fn=None, cv=None):
    """
    Ranking of features using cross-validation

    Parameters
    ----------
    feat_df : pandas.DataFrame
        features data only
    y : numpy.array_like
        target/output values corresponding to feat_df
    model : sklearn.BaseEstimator
        model for feature evaluation (default = LinearRegression)
    score_fn : function in form score = score_fn(y_true, y_pred)
        a model score function in the form of a sklearn metric (eg RMSE) to evaluate model (default = -RMSE)
    cv : int
        number of cross-validated folds to use (default = )

    Returns
    -------
    feat_scores : list
        list of score dicts corresponding to features in feat_df
    """
    logger.info('Ranking: ')
    if score_fn is None:
        logger.info('Using negative RMSE score')
    else:
        logger.info('Using user score')

    feat_scores = []
    for i, (feat_key, feat_vec) in enumerate(feat_df.iteritems()):
        logger.info('Scoring feature {0} of {1}'.format(i+1, feat_df.shape[1]))

        scores, predicted = score_model(pd.DataFrame(feat_vec), y, model=model, score_fn=score_fn, cv=cv, find_predicted=False)

        if score_fn is None:
            score = -np.sqrt((scores['test_-RMSE']**2).mean())
        else:
            score = scores['test_user'].mean()
        feat_scores.append(score)

    feat_scores = np.array(feat_scores)

    logger.info('Best score: {0}'.format(feat_scores.max()))
    logger.info('Best feat: {0}'.format(feat_df.columns[np.argmax(feat_scores)]))
    return feat_scores

def score_model(feat_df, y, model=linear_model.LinearRegression(), score_fn=None,
                cv=None, find_predicted=True, print_scores=False):
    """
    Cross-validated model score from ground truth data

    Parameters
    ----------
    feat_df : pandas.DataFrame
        features data only
    y : numpy.array_like
        target/output values corresponding to feat_df
    model : sklearn.BaseEstimator
        model for feature evaluation (default = LinearRegression)
    score_fn : function in form score = score_fn(y_true, y_pred)
        a model score function in the form of a sklearn metric (eg RMSE) to evaluate model (default = -RMSE)
    cv : int
        number of cross-validated folds to use (default = leave one out)
    find_predicted : bool
        return the predicted outputs (default = True)
    print_scores : bool
        log scores to consolde (default = False)

    Returns
    -------
    scores: dict
    predicted: array_like
        predicted outputs (find_predicted = True)
    """


    predicted = None
    if cv is None:
        cv = len(y)        # Leave one out

    if score_fn is not None:
        scoring = {#'R2': make_scorer(metrics.r2_score),        # R2 in cross validation is suspect
                   '-RMSE': make_scorer(lambda meas, pred: -np.sqrt(metrics.mean_squared_error(meas, pred))),
                   'user': make_scorer(score_fn)}
    else:
        scoring = {'-RMSE': make_scorer(lambda meas, pred: -np.sqrt(metrics.mean_squared_error(meas, pred)))}

    scores = cross_validate(model, feat_df, y, scoring=scoring, cv=cv, n_jobs=-1)

    if print_scores:
        rmse_ci = np.percentile(-scores['test_-RMSE'], [5, 95])
        logger.info('RMSE mean: {0:.4f}, std: {1:.4f}, 5-95%: {2:.4f} - {3:.4f}'.format(-scores['test_-RMSE'].mean(),
                scores['test_-RMSE'].std(), rmse_ci[0], rmse_ci[1]))

    if find_predicted:
        predicted = cross_val_predict(model, feat_df, y, cv=cv)  #)
        scores['R2_stacked'] = metrics.r2_score(y, predicted)   # Also suspect for validation, but better than cross validated R2
        if print_scores:
            logger.info('R2 (stacked): {0:.4f}'.format(scores['R2_stacked']))
    return scores, predicted
