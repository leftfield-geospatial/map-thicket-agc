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
from matplotlib import pyplot
from matplotlib import patches
from scipy import stats as stats
from sklearn import linear_model
import pandas as pd
from map_thicket_agc import feature_selection

logger = get_logger(__name__)

def scatter_ds(data, x_col=None, y_col=None, class_col=None, label_col=None, thumbnail_col=None, do_regress=True,
               x_label=None, y_label=None, xfn=lambda x: x, yfn=lambda y: y):
    """
    2D scatter plot of pandas dataframe with annotations etc

    Parameters
    ----------
    data : pandas.DataFrame
    x_col : str
        column to use for x axis
    y_col : str
        column to use for y axis
    class_col : str
        column to use for colouring classes (optional)
    label_col : str
        column to use for text labels of data points (optional)
    thumbnail_col : str
        columnt to use for image thumbnails (optional)
    do_regress : bool
        display regression accuracies (default = False)
    x_label : str
        text string for x axis label, (default = None uses x_col)
    y_label : str
        text string for y axis label, (default = None uses y_col)
    xfn : function
        function for modifying x data (e.g. lambda x: np.log10(x)) (optional)
    yfn : function
        function for modifying y data (e.g. lambda x: np.log10(x)) (optional)

    Returns
    -------
    R2, RMSE statistics
    """

    ims = 20.       # scale factor for thumbnails
    if x_col is None:
        x_col = data.columns[0]
    if y_col is None:
        y_col = data.columns[1]

    x = xfn(data[x_col])
    y = yfn(data[y_col])

    if class_col is None:
        data['class_col'] = np.zeros((data.shape[0],1))
        class_col = 'class_col'

    classes = np.array([class_name for class_name, class_group in data.groupby(by=class_col)])
    n_classes =  len(classes)
    if n_classes == 1:
        colours = [(0., 0., 0.)]
    else:
        colours = ['tab:orange', 'g', 'r', 'b', 'y', 'k', 'm']

    xlim = [np.nanmin(x), np.nanmax(x)]
    ylim = [np.nanmin(y), np.nanmax(y)]
    xd = np.diff(xlim)[0]
    yd = np.diff(ylim)[0]

    pyplot.axis('tight')
    pyplot.axis(xlim + ylim)
    ax = pyplot.gca()
    handles = [0] * n_classes

    # loop through data grouped by class (strata) if any
    for class_i, (class_label, class_data) in enumerate(data.groupby(by=class_col)):
        colour = colours[class_i]
        if thumbnail_col is None:   # plot data points
            pyplot.plot(xfn(class_data[x_col]), yfn(class_data[y_col]), markerfacecolor=colour, marker='.', label=class_label, linestyle='None',
                       markeredgecolor=colour, markersize=5)
        for rowi, row in class_data.iterrows():
            xx = xfn(row[x_col])
            yy = yfn(row[y_col])
            if label_col is not None:   # add a text label for each point
                pyplot.text(xx - .0015, yy - .0015, row[label_col],
                           fontdict={'size': 9, 'color': colour, 'weight': 'bold'})

            if thumbnail_col is not None:   # add a thumbnail for each point
                imbuf = np.array(row[thumbnail_col])
                band_idx = [0, 1, 2]
                if imbuf.shape[2] == 8:  # wv3
                    band_idx = [4, 2, 1]
                elif imbuf.shape[2] >= 8:  # wv3 + pan
                    band_idx = [5, 3, 2]

                extent = [xx - xd/(2 * ims), xx + xd/(2 * ims), yy - yd/(2 * ims), yy + yd/(2 * ims)]
                pyplot.imshow(imbuf[:, :, band_idx], extent=extent, aspect='auto')
                handles[class_i] = ax.add_patch(patches.Rectangle((extent[0], extent[2]), xd/ims, yd/ims,
                                                fill=False, edgecolor=colour, linewidth=2.))

    if do_regress:  # find and display regression error stats
        (slope, intercept, r, p, stde) = stats.linregress(x, y)
        scores, predicted = feature_selection.score_model(x.to_numpy().reshape(-1,1), y.to_numpy().reshape(-1,1), model=linear_model.LinearRegression(),
                                                        find_predicted=True, cv=len(x), print_scores=False, score_fn=None)

        pyplot.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(scores['R2_stacked'], 2)),
                   fontdict={'size': 12})
        yr = np.array(xlim)*slope + intercept
        pyplot.plot(xlim, yr, 'k--', lw=2, zorder=-1)

        yhat = x * slope + intercept
        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        logger.info('Regression scores')
        logger.info('RMSE: {0:.4f}'.format(rmse))
        logger.info('RMSE LOOCV: {0:.4f}'.format(-scores['test_-RMSE'].mean()))
        logger.info('R^2:  {0:.4f}'.format(r ** 2))
        logger.info('R^2 stacked: {0:.4f}'.format(scores['R2_stacked']))
        logger.info('P (slope=0): {0:.4f}'.format(p))
        logger.info('Slope: {0:.4f}'.format(slope))
        logger.info('Std error of slope: {0:.4f}'.format(stde))
    else:
        r = np.nan
        rmse = np.nan

    if x_label is not None:
        pyplot.xlabel(x_label, fontdict={'size': 12})
    else:
        pyplot.xlabel(x_col, fontdict={'size': 12})

    if y_label is not None:
        pyplot.ylabel(y_label, fontdict={'size': 12})
    else:
        pyplot.ylabel(y_col, fontdict={'size': 12})

    if n_classes > 1:
        if not thumbnail_col is None:
            pyplot.legend(handles, classes, fontsize=12)
        else:
            pyplot.legend(classes, fontsize=12)
    pyplot.pause(0.1)
    return r ** 2, rmse


def scatter_y_actual_vs_pred(y, pred, scores, xlabel='Measured AGC (t C ha$^{-1}$)', ylabel='Predicted AGC (t C ha$^{-1}$)'):
    """
    Scatter plot of predicted vs actual data in format for reporting

    Parameters
    ----------
    y : numpy.array_like
        actual data
    pred : numpy.array_like
        predicted data
    scores : dict
        scores dict as returned by feature_selection.score_model()
    xlabel : str
        label for x axis (optional)
    ylabel : str
        label for y axis (optional)
    """

    df = pd.DataFrame({xlabel: y, ylabel: pred})    # form a datafram for scatter_ds
    scatter_ds(df, do_regress=False)

    mn = np.min([y, pred], initial=0)
    mx = np.max([y, pred])
    h, = pyplot.plot([mn, mx], [mn, mx], 'k--', lw=2, zorder=-1, label='1:1')
    pyplot.xlim(0, mx)
    pyplot.ylim(0, mx)
    pyplot.text(26, 5, str.format('$R^2$ = {0:.2f}', scores['R2_stacked']),
               fontdict={'size': 11})
    pyplot.text(26, 2, str.format('RMSE = {0:.2f} t C ha{1}',np.abs(scores['test_-RMSE']).mean(),'$^{-1}$'),
               fontdict={'size': 11})
    pyplot.tight_layout()
    pyplot.legend([h], ['1:1'], frameon=False)
    pyplot.pause(0.1)   # hack to get around pyplot bug when saving figure
