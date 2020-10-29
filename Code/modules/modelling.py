# TODO remove future etc
# TODO replace GladImageReader with rasterio or equivalent
# TODO replace GladVectorReader with ?? - something that reads to pandas?



from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import sys
import warnings, logging
import gdal
import ogr
import numpy as np
import osr
import pylab
from rasterio.rio.options import nodata_opt
from scipy import stats as stats
import scipy.signal as signal
from matplotlib import patches
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict, cross_validate

import collections
from collections import OrderedDict
from sklearn.preprocessing import PolynomialFeatures
# from pandas import DataFrame as pd
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import matplotlib.pyplot as pyplot
import geopandas as gpd, pandas as pd

# Python Imaging Library imports
from PIL import Image
from PIL import ImageDraw
import os
import numpy as np
import rasterio
import re
import rasterio
from rasterio.features import sieve
from rasterio.windows import Window
from rasterio.mask import raster_geometry_mask
if sys.version_info.major == 3:
    from sklearn.metrics import make_scorer
else:
    from sklearn.metrics.scorer import make_scorer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def nanentropy(x, axis=None):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """
    if axis is None:
        # quantise x
        nbins = 100
        x = x[~np.isnan(x)]
        if len(x) < 10:
            return 0.
        x = x-x.min()
        x = np.int32(np.round(np.float32(nbins*x)/x.max()))

        value, counts = np.unique(x, return_counts=True)
        p = np.array([count/float(x.size) for count in counts])
        # p = np.array([np.size(x[x == i]) / (1.0 * x.size) for i in np.unique(x)])
        # compute Shannon entropy in bits
        return -np.sum(p * np.log2(p))
    else:        # hack for 2D slices of 3D array (on a rolling_window 3D array)
        along_axis = np.setdiff1d(range(0, len(x.shape)), axis)[0]
        e = np.zeros((x.shape[along_axis]))
        for slice in range(x.shape[along_axis]):
            if along_axis == 0:
                xa = x[slice,:,:]
            elif along_axis == 1:
                xa = x[:,slice,:]
            elif along_axis == 2:
                xa = x[:,:,slice]
            e[slice] = nanentropy(xa)
        return e


def entropy(x, along_axis=None):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """
    # quantise x
    nbins = 100
    x = x.flatten()
    x = x-x.min()
    x = np.int32(np.round(old_div((nbins*x),x.max())))

    p = np.array([old_div(np.size(x[x == i]), (1.0 * x.size)) for i in np.unique(x)])
    # compute Shannon entropy in bits
    return -np.sum(p * np.log2(p))


def scatter_y_actual_vs_pred(y, pred, scores, xlabel='Measured AGC (t C ha$^{-1}$)', ylabel='Predicted AGC (t C ha$^{-1}$)'):
    df = pd.DataFrame({xlabel: y, ylabel: pred})
    scatter_ds(df, do_regress=False)
    # scatter_plot(y, pred, xlabel=xlabel, ylabel=ylabel, do_regress = False)

    mn = np.min([y, pred])
    mx = np.max([y, pred])
    h, = pyplot.plot([0, mx], [0, mx], 'k--', lw=2, zorder=-1, label='1:1')
    pyplot.xlim(0, mx)
    pyplot.ylim(0, mx)
    pyplot.text(26, 5, str.format('$R^2$ = {0:.2f}', scores['R2_stacked']),
               fontdict={'size': 11})
    pyplot.text(26, 2, str.format('RMSE = {0:.2f} t C ha{1}',np.abs(scores['test_-RMSE']).mean(),'$^{-1}$'),
               fontdict={'size': 11})
    pyplot.show()
    pyplot.tight_layout()
    pyplot.legend([h], ['1:1'], frameon=False)
    pyplot.pause(0.1)


def scatter_ds(data, x_col=None, y_col=None, class_col=None, label_col=None, thumbnail_col=None, do_regress=True,
               x_label=None, y_label=None, xfn=lambda x: x, yfn=lambda y: y):
    """

    Parameters
    ----------
    data
    x_col
    y_col
    class_col
    label_col
    thumbnail_col
    do_regress
    x_label
    y_label
    xfn
    yfn

    Returns
    -------

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

    # TO DO: either remove thumbnail option or refactor
    # TO DO: sort classes
    # if 'Intact' in classes and 'Moderate' in classes and 'Degraded' in classes and classes.size==3:
    #     classes = np.array(['Degraded', 'Moderate', 'Intact'])
    classes = np.array([class_name for class_name, class_group in data.groupby(by=class_col)])
    n_classes =  len(classes)
    if n_classes == 1:
        colours = [(0., 0., 0.)]
    else:
        colours = ['tab:orange', 'g', 'r', 'b', 'y', 'k', 'm']

    xlim = [x.min(), x.max()]
    ylim = [y.min(), y.max()]
    xd = np.diff(xlim)[0]
    yd = np.diff(ylim)[0]

    pyplot.axis('tight')
    pyplot.axis(xlim + ylim)
    ax = pyplot.gca()
    handles = [0] * n_classes

    for class_i, (class_label, class_data) in enumerate(data.groupby(by=class_col)):
        colour = colours[class_i]
        if thumbnail_col is None:
            pylab.plot(xfn(class_data[x_col]), yfn(class_data[y_col]), markerfacecolor=colour, marker='.', label=class_label, linestyle='None',
                       markeredgecolor=colour, markersize=5)
        for rowi, row in class_data.iterrows():
            xx = xfn(row[x_col])
            yy = yfn(row[y_col])
            if label_col is not None:   # add a text label for each point
                pylab.text(xx - .0015, yy - .0015, row[label_col],
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
        scores, predicted = FeatureSelector.score_model(x[:,None], y[:,None], model=linear_model.LinearRegression(),
                                                        find_predicted=True, cv=len(x), print_scores=False, score_fn=None)

        pylab.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(scores['R2_stacked'], 2)),
                   fontdict={'size': 12})
        yr = np.array(xlim)*slope + intercept
        pyplot.plot(xlim, yr, 'k--', lw=2, zorder=-1)

        yhat = x * slope + intercept
        rmse = np.sqrt(np.mean((y - yhat) ** 2))

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
        pylab.xlabel(x_col[-1], fontdict={'size': 12})

    if y_label is not None:
        pyplot.ylabel(y_label, fontdict={'size': 12})
    else:
        pylab.ylabel(y_col[-1], fontdict={'size': 12})

    if n_classes > 1:
        if not thumbnail_col is None:
            pylab.legend(handles, classes, fontsize=12)
        else:
            pylab.legend(classes, fontsize=12)
    pylab.show()
    return r ** 2, rmse

def scatter(x, y, class_labels=None, labels=None, thumbnails=None, do_regress=True,
               x_label=None, y_label=None, xfn=lambda x: x, yfn=lambda y: y):
    """

    Parameters
    ----------
    data
    x_col
    y_col
    class_col
    label_col
    thumbnail_col
    do_regress
    x_label
    y_label
    xfn
    yfn

    Returns
    -------

    """
    ims = 20.       # scale factor for thumbnails

    x = xfn(x)
    y = yfn(y)

    if class_labels is None:
        class_labels = pd.DataFrame({'class_col':np.zeros((x.shape[0],1))})
        class_col = 'class_col'

    # TO DO: either remove thumbnail option or refactor
    # TO DO: sort classes
    # if 'Intact' in classes and 'Moderate' in classes and 'Degraded' in classes and classes.size==3:
    #     classes = np.array(['Degraded', 'Moderate', 'Intact'])
    classes = np.unique(class_labels)   # np.array([class_name for class_name, class_group in class_labels.groupby(by=class_col)])
    n_classes =  len(classes)
    if n_classes == 1:
        colours = [(0., 0., 0.)]
    else:
        colours = ['g', 'tab:orange', 'r', 'b', 'y', 'k', 'm']

    xlim = [x.min(), x.max()]
    ylim = [y.min(), y.max()]
    xd = np.diff(xlim)[0]
    yd = np.diff(ylim)[0]

    pyplot.axis('tight')
    pyplot.axis(xlim + ylim)
    ax = pyplot.gca()
    handles = [0] * n_classes

    # for class_i, (class_label, class_data) in enumerate(class_labels.groupby(by=class_col)):
    for class_i, class_label in enumerate(classes):
        class_idx = class_labels == class_label
        colour = colours[class_i]
        y_i = y[class_idx]
        x_i = x[class_idx]
        label_i = labels[class_idx]

        if thumbnails is None:
            pylab.plot(x_i, y_i, markerfacecolor=colour, marker='.', label=class_label, linestyle='None',
                       markeredgecolor=colour)
        for rowi, row in class_data.iterrows():
            xx = xfn(row[x_col])
            yy = yfn(row[y_col])
            if label_col is not None:   # add a text label for each point
                pylab.text(xx - .0015, yy - .0015, row[label_col],
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
                handles[class_i] = ax.add_patch(patches.Rectangle(extent[0], extent[2], xd/ims, yd/ims,
                                                fill=False, edgecolor=colour, linewidth=2.))

    if do_regress:  # find and display regression error stats
        (slope, intercept, r, p, stde) = stats.linregress(x, y)
        scores, predicted = FeatureSelector.score_model(x[:,None], y[:,None], model=linear_model.LinearRegression(),
                                                        find_predicted=True, cv=len(x), print_scores=True)

        pylab.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(scores['R2_stacked'], 2)),
                   fontdict={'size': 12})
        yr = np.array(xlim)*slope + intercept
        pyplot.plot(xlim, yr, 'k--', lw=2, zorder=-1)

        yhat = x * slope + intercept
        rmse = np.sqrt(np.mean((y - yhat) ** 2))

        logger.info('RMSE = {0:.4f}'.format(rmse))
        logger.info('LOOCV RMSE = {0:.4f}'.format(np.sqrt(-scores['test_user'].mean())))
        logger.info('R^2  = {0:.4f}'.format(r ** 2))
        logger.info('Stacked R^2  = {0:.4f}'.format(scores['R2_stacked']))
        logger.info('P (slope=0) = {0:f}'.format(p))
        logger.info('Slope = {0:.4f}'.format(slope))
        logger.info('Std error of slope = {0:.4f}'.format(stde))
    else:
        r = np.nan
        rmse = np.nan

    if x_label is not None:
        pyplot.xlabel(x_label, fontdict={'size': 12})
    else:
        pylab.xlabel(x_col, fontdict={'size': 12})

    if y_label is not None:
        pyplot.ylabel(y_label, fontdict={'size': 12})
    else:
        pylab.xlabel(y_col, fontdict={'size': 12})

    if n_classes > 1:
        if not thumbnail_col is None:
            pylab.legend(handles, classes, fontsize=12)
        else:
            pylab.legend(classes, fontsize=12)
    return r ** 2, rmse

def scatter_plot(x, y, class_labels=None, labels=None, thumbnails=None, do_regress=True, xlabel=None, ylabel=None,
                 xfn=lambda x: x, yfn=lambda y: y):
    # x = np.array([xfn(plot[x_feat_key]) for plot in im_feat_dict.values()])
    if type(x[0]) is np.ndarray:  # this is pixel data and requires concat to flatten it
        cfn = lambda x: np.hstack(x)[::5]
        thumbnails = None
    else:
        cfn = lambda x: x

    # if xfn is not None:
    #     x = xfn(x)
    # y = np.array([yfn(plot[y_feat_key]) for plot in im_feat_dict.values()])
    # if type(x[0]) is np.ndarray:
    #     ycfn = lambda x: np.concatenate(x)
    # else:
    #     ycfn = lambda x: x

    # if yfn is not None:
    #     y = yfn(y)

    # if show_class_labels == True:
    #     class_labels = np.array([plot[class_key] for plot in im_feat_dict.values()])
    # else:
    #     class_labels = np.zeros(x.__len__())
    # if show_thumbnails == True:
    #     thumbnails = np.array([plot['thumbnail'] for plot in im_feat_dict.values()])
    #
    # if show_labels == True:
    #     labels = np.array([plot['ID'] for plot in im_feat_dict.values()])
    if class_labels is None:
        class_labels = np.zeros(x.__len__())
    classes = np.unique(class_labels)
    if 'Intact' in classes and 'Moderate' in classes and 'Degraded' in classes and classes.size==3:
        classes = np.array(['Degraded', 'Moderate', 'Intact'])

    if classes.__len__() == 1:
        # colours = [u'#1f77b4']
        colours = [(0., 0., 0.)]
    else:
        # colours = ['r', 'm', 'b', 'g', 'y', 'k', 'o']
        colours = ['r', 'tab:orange', 'g', 'b', 'y', 'k', 'm']

    ylim = [np.min(cfn(y)), np.max(cfn(y))]
    xlim = [np.min(cfn(x)), np.max(cfn(x))]
    xd = np.diff(xlim)[0]
    yd = np.diff(ylim)[0]

    # pylab.figure()
    pylab.axis('tight')
    pylab.axis(np.concatenate([xlim, ylim]))
    # pylab.hold('on')
    ax = pylab.gca()
    handles = np.zeros(classes.size).tolist()
    #

    for ci, (class_label, colour) in enumerate(zip(classes, colours[:classes.__len__()])):
        class_idx = class_labels == class_label
        if thumbnails is None:
            pylab.plot(cfn(x[class_idx]), cfn(y[class_idx]), markerfacecolor=colour, marker='.', label=class_label, linestyle='None',
                       markeredgecolor=colour)

        for xyi, (xx, yy) in enumerate(zip(x[class_idx], y[class_idx])):  # , np.array(plot_names)[class_idx]):
            if type(xx) is np.ndarray:
                xx = xx[0]
            if type(yy) is np.ndarray:
                yy = yy[0]
            if not labels is None:
                pylab.text(xx - .0015, yy - .0015, np.array(labels)[class_idx][xyi],
                           fontdict={'size': 9, 'color': colour, 'weight': 'bold'})

            if not thumbnails is None:
                imbuf = np.array(thumbnails)[class_idx][xyi]
                band_idx = [0, 1, 2]
                if imbuf.shape[2] == 8:  # guess wv3
                    band_idx = [4, 2, 1]

                ims = 20.
                extent = [xx - old_div(xd, (2 * ims)), xx + old_div(xd, (2 * ims)), yy - old_div(yd, (2 * ims)), yy + old_div(yd, (2 * ims))]
                # pylab.imshow(imbuf[:, :, :3], extent=extent, aspect='auto')  # zorder=-1,
                pylab.imshow(imbuf[:, :, band_idx], extent=extent, aspect='auto')  # zorder=-1,
                handles[ci] = ax.add_patch(
                    patches.Rectangle((xx - old_div(xd, (2 * ims)), yy - old_div(yd, (2 * ims))), old_div(xd, ims), old_div(yd, ims), fill=False,
                                      edgecolor=colour, linewidth=2.))
                # pylab.plot(mPixels[::step], dRawPixels[::step], color='k', marker='.', linestyle='', markersize=.5)
        if do_regress and classes.__len__() > 1 and False:
            (slope, intercept, r, p, stde) = stats.linregress(cfn(x[class_idx]), cfn(y[class_idx]))
            pylab.text(xlim[0] + xd * 0.7, ylim[0] + yd * 0.05 * (ci + 2),
                       '{1}: $R^2$ = {0:.2f}'.format(np.round(r ** 2, 2), classes[ci]),
                       fontdict={'size': 10, 'color': colour})


    if do_regress:
        (slope, intercept, r, p, stde) = stats.linregress(cfn(x), cfn(y))
        scores, predicted = FeatureSelector.score_model(cfn(x)[:,None], cfn(y)[:,None], model=linear_model.LinearRegression(), find_predicted=True, cv=len(x), print_scores=True)
        r2_stacked = scores['R2_stacked']
        rmse = np.abs(scores['test_user']).mean()

        pylab.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(scores['R2_stacked'], 2)),
                   fontdict={'size': 12})
        logger.info('RMSE^2 = {0:.4f}'.format(rmse))
        logger.info('R^2 = {0:.4f}'.format(r ** 2))
        logger.info('P (slope=0) = {0:f}'.format(p))
        logger.info('Slope = {0:.4f}'.format(slope))
        logger.info('Std error of slope = {0:.4f}'.format(stde))
        yhat = cfn(x) * slope + intercept
        rmse = np.sqrt(np.mean((cfn(y) - yhat) ** 2))
        logger.info('RMS error = {0:.4f}'.format(rmse))

        yr = [0, 0]
        yr[0] = xlim[0]*slope + intercept
        yr[1] = xlim[1]*slope + intercept
        pylab.plot(xlim, yr, 'k--', lw=2, zorder=-1)
    else:
        r = np.nan
        rmse = np.nan

    if xlabel is not None:
        pylab.xlabel(xlabel, fontdict={'size': 12})
    # else:
    #     pylab.xlabel(x_feat_key, fontdict={'size': 12})
    if ylabel is not None:
        pylab.ylabel(ylabel, fontdict={'size': 12})
    # else:
    #     pylab.ylabel(y_feat_key, fontdict={'size': 12})
    # pylab.ylabel(yf)
    # pylab.grid()
    if classes.size > 1:
        if not thumbnails is None:
            pylab.legend(handles, classes, fontsize=12)
        else:
            pylab.legend(classes, fontsize=12)
    return r ** 2, rmse


    @staticmethod
    def plot_clf(clf, X, y, feat_keys=None, class_labels=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.colors import ListedColormap

        if X.shape[1] > 2:
            raise Exception('X.shape[1] > 2')
        if X.shape[1] > 2:
            raise Exception('X.shape[1] > 2')
        h = 0.002
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.Set2
        plt.figure()
        ax = plt.subplot(1, 1, 1)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        # if hasattr(clf, "decision_function"):
        #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # else:
        #     # Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        # ax.contourf(xx, yy, Z, cmap=cm, alpha=1, levels=[0,1,2,3,4,5])
        ax.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=cm, aspect='auto')

        # Plot the training points
        s = 100
        colours = cm(np.linspace(0., 1., np.unique(y).size))  #make same colours as contourf
        h = []
        for i, class_id in enumerate(np.unique(y)):
            class_idx = y == class_id
            h.append(ax.scatter(X[class_idx, 0], X[class_idx, 1], color=colours[i], edgecolor='black', s=20))

        # ax.scatter(X[::s, 0], X[::s, 1], c=y[::s], cmap=cm, edgecolor='black', s=15)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        if feat_keys is not None:
            ax.set_xlabel(feat_keys[0])
            ax.set_ylabel(feat_keys[1])

        # from matplotlib.lines import
        # custom_lines = [Line2D([0], [0], color=cm(0.), lw=4),
        #                 Line2D([0], [0], color=cm(.5), lw=4),
        #                 Line2D([0], [0], color=cm(1.), lw=4)]

        # fig, ax = plt.subplots()
        # lines = ax.plot(data)
        if class_labels is not None:
            ax.legend(h, class_labels)
        # ax.set_xticks(())
        # ax.set_yticks(())

# class GdalImageReader(object):
#     def __init__(self, file_name):
#         self.file_name = file_name
#         if not os.path.exists(file_name):
#             raise Exception("File does not exist: {0}".format(file_name))
#         self.ds = None
#         self.__open()
#
#     def __open(self):
#         self.ds = gdal.OpenEx(self.file_name, gdal.OF_RASTER)
#         if self.ds is None:
#             raise Exception("Could not open {0}".format(self.file_name))
#
#         logger.info('Driver: {0}'.format(self.ds.GetDriver().LongName))
#         self.width = self.ds.RasterXSize
#         self.height = self.ds.RasterYSize
#         self.num_bands = self.ds.RasterCount
#         logger.info('Size: {0} x {1} x {2} (width x height x bands)'.format(self.ds.RasterXSize, self.ds.RasterYSize, self.ds.RasterCount))
#         logger.info('Projection: {0}'.format(self.ds.GetProjection()))
#         self.spatial_ref = osr.SpatialReference(self.ds.GetProjection())
#         self.geotransform = self.ds.GetGeoTransform()
#         if not self.geotransform is None:
#             self.origin = np.array([self.geotransform[0], self.geotransform[3]])
#             logger.info('Origin = ({0}, {1})'.format(self.geotransform[0], self.geotransform[3]))
#             logger.info('Pixel Size =  = ({0}, {1})'.format(self.geotransform[0], self.geotransform[3]))
#             logger.info('Pixel Size = ({0}, {1})'.format(self.geotransform[1], self.geotransform[5]))
#             self.pixel_size = np.array([self.geotransform[1], self.geotransform[5]])
#         else:
#             self.origin = np.array([0, 0])
#             self.pixel_size = np.array([1, 1])
#         self.gdal_dtype = self.ds.GetRasterBand(1).DataType
#         if self.gdal_dtype == gdal.GDT_UInt16:
#             self.dtype = np.uint16
#         elif self.gdal_dtype == gdal.GDT_Int16:
#             self.dtype = np.int16
#         if self.gdal_dtype == gdal.GDT_Float32:
#             self.dtype = np.float32
#         elif self.gdal_dtype == gdal.GDT_Float64:
#             self.dtype = np.float64
#         else:
#             self.dtype = np.float32
#
#         self.block_size = self.ds.GetRasterBand(1).GetBlockSize()
#         self.image_array = None
#
#     def cleanup(self):
#         self.image_array = None
#         self.ds = None
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.cleanup()
#     def __del__(self):
#         self.cleanup()
#
#     def world_to_pixel(self, x, y):
#         row = old_div((y - self.origin[1]),self.pixel_size[1])     # row
#         col =  old_div((x - self.origin[0]),self.pixel_size[0])    # col
#         return (col, row)
#
#     def pixel_to_world(self, col, row):
#         y = row * self.pixel_size[1] + self.origin[1]
#         x = col * self.pixel_size[0] + self.origin[0]
#         return (col, row)
#
#     def read_image(self):
#         # the below orders pixels by band, row, col but we want band as last dimension
#         # self.image_array = self.ds.ReadAsArray(buf_type=self.gdal_dtype)
#         self.image_array = self.read_image_roi()
#         return self.image_array
#
#     def read_image_roi(self, col_range=None, row_range=None, band_range=None):
#         if row_range is None:
#             row_range = [0, self.height]
#         if col_range is None:
#             col_range = [0, self.width]
#         if band_range is None:
#             band_range = [0, self.num_bands]
#
#         # check ranges
#         for drange, drange_max in zip([row_range, col_range, band_range], [self.height, self.width, self.num_bands]):
#             drange[0] = np.maximum(0, drange[0])
#             drange[1] = np.minimum(drange_max, drange[1])
#
#         image_roi = np.zeros((np.diff(row_range)[0], np.diff(col_range)[0], np.diff(band_range)[0]), dtype=self.dtype)
#         for bi in range(band_range[0], band_range[1]):
#             image_roi[:, :, bi] = self.ds.GetRasterBand(bi + 1).ReadAsArray(int(col_range[0]), int(row_range[0]), int(np.diff(col_range)[0]),
#                                                                  int(np.diff(row_range)[0]), buf_type=self.gdal_dtype)
#         return image_roi

class GdalVectorReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        if not os.path.exists(file_name):
            raise Exception("File does not exist: {0}".format(file_name))
        self.ds = None
        self.__open()

    def __open(self):
        self.ds = gdal.OpenEx(self.file_name, gdal.OF_VECTOR)
        if self.ds is None:
            raise Exception("Could not open {0}".format(self.file_name))
        self.num_layers = self.ds.GetLayerCount()
        self.layer_dict = {}

    def cleanup(self):
        self.layer_dict = None
        self.ds = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()

    def read(self, id_field = None):
        self.ds.ResetReading()
        for li in range(0, self.num_layers):
            layer = self.ds.GetLayerByIndex(li)
            # self.layers.append(layer)
            layer.ResetReading()
            feat_defn = layer.GetLayerDefn()
            feat_dict = {}
            logger.info('Reading feats in layer: {0}'.format(layer.GetName()))
            for feati, feat in enumerate(layer):
                # print '.',
                fdict = {}
                for fieldi in range(feat_defn.GetFieldCount()):
                    field_defn = feat_defn.GetFieldDefn(fieldi)
                    fdict[field_defn.GetName()] = feat.GetField(fieldi)

                    # find id field if it exists otherwise set to feature index
                    if id_field is None:
                        idFields = ['id', 'ID', 'Id', 'Name', 'name']
                        id = None
                        for idField in idFields:
                            if idField in list(fdict.keys()):
                                id = fdict[idField]
                                break
                    else:
                        id = fdict[id_field]

                    if id is None:
                        id = str(feati)
                        fdict['ID'] = id

                    if False:
                        id = fdict['ID']
                        if id[0] == 'S' or id[:3] == 'TCH':
                            fdict['DegrClass'] = 'Severe'
                        elif id[0] == 'M':
                            fdict['DegrClass'] = 'Moderate'
                        elif id[0] == 'P' or id[:3] == 'INT':
                            fdict['DegrClass'] = 'Pristine'
                        else:
                            fdict['DegrClass'] = '?'

                geom = feat.GetGeometryRef()
                fdict['geom'] = geom.Clone()
                if geom is not None and (geom.GetGeometryType() == ogr.wkbPolygon):
                    logger.info("%s Polygon with %d points"  % (id, geom.GetGeometryRef(0).GetPointCount()))
                    fdict['points'] = geom.GetGeometryRef(0).GetPoints()[:-1]
                    # pixCnr = []
                    # for point in f['points']:
                    #     pixCnr.append(World2Pixel(geotransform, point[0], point[1]))
                elif geom is not None and (geom.GetGeometryType() == ogr.wkbPoint or geom.GetGeometryType() == ogr.wkbPoint25D or geom.GetGeometryType() == ogr.wkbPointZM):
                    logger.info("%s Point (%.6f, %.6f)" % (id, geom.GetX(), geom.GetY()))
                    fdict['X'] = geom.GetX()
                    fdict['Y'] = geom.GetY()
                    if False:    #'GNSS_Heigh' in fdict.keys():
                        fdict['Z'] = fdict['GNSS_Heigh']  # ? - should be able to get this from geom
                    else:
                        fdict['Z'] = geom.GetZ()  # this has been xformed from elippsoidal to Sa Geoid 2010
                    # f['ID'] = f['Datafile'][:-4] + str(f['Point_ID'])
                else:
                    logger.warning("Unknown geometry")
                feat_dict[id] = fdict
                # print ' '

            self.layer_dict[layer.GetName()]={'feat_dict':feat_dict, 'spatial_ref':layer.GetSpatialRef()}
        return self.layer_dict

from enum import Enum

class GroundClass(Enum):
    Ground = 1
    DPlant = 2
    LPlant = 3
    Shadow = 4

# class to extract features from polygons in an raster
class ImPlotFeatureExtractor(object):
    def __init__(self, image_reader=rasterio.io.DatasetReader, plot_feat_dict={}, plot_data_gdf=gpd.GeoDataFrame()):
        ''' Class
        Parameters
        ----------
        image_reader
        plot_feat_dict
        plot_data_gdf
        '''
        self.image_reader = image_reader
        self.plot_data_gdf = plot_data_gdf
        self.im_plot_data_gdf = gpd.GeoDataFrame()


    @staticmethod
    def get_band_info(num_bands=9):
        if num_bands == 8:  # assume Wv3
            # pan_bands = np.array([1, 2, 3, 4, 5, 6])     # this needs some experimentation and may be better using the actual pan info
            pan_bands = [1, 2, 4, 5]
            band_dict = OrderedDict([('C', 0), ('B', 1), ('G', 2), ('Y', 3), ('R', 4), ('RE', 5), ('NIR', 6), ('NIR2', 7)])
            # band_dict = OrderedDict(
            #     [('B', 1), ('G', 2), ('Y', 3), ('R', 4), ('RE', 5), ('NIR', 6), ('NIR2', 7)])       # exclude C which is noisy
        elif num_bands == 9:  # assume Wv3 ms + pan in band 0
            # pan_bands = np.array([0])
            pan_bands = [2, 3, 5, 6]
            band_dict = OrderedDict(
                [('C', 1), ('B', 2), ('G', 3), ('Y', 4), ('R', 5), ('RE', 6), ('NIR', 7), ('NIR2', 8)])
            # band_dict = OrderedDict(
            #     [('B', 2), ('G', 3), ('Y', 4), ('R', 5), ('RE', 6), ('NIR', 7), ('NIR2', 8)])       # exclude C which is noisy
        else:  # assume NGI
            # pan_bands = np.array([0, 1, 2, 3])
            pan_bands = [0, 1, 2, 3]
            band_dict = OrderedDict([('R', 0), ('G', 1), ('B', 2), ('NIR', 3)])
        return pan_bands, band_dict

    @staticmethod
    def extract_patch_ms_features_ex(imbuf, mask=None, per_pixel=False, include_texture=True):
        feat_dict = OrderedDict()

        # note that ms bands have been scaled and likely will not sum back to approx pan, so it is better to use the actual pan
        pan_bands, band_dict = ImPlotFeatureExtractor.get_band_info(imbuf.shape[2])

        if mask is None:
            mask = np.all(imbuf>0, axis=2)  #np.ones(imbuf.shape[:2])
        mask = np.bool8(mask)
        # mask = mask & np.all(imbuf > 0, axis=2) & np.all(imbuf[:,:,], axis=2)
        imbuf_mask = np.ndarray(shape=(np.int32(mask.sum()), imbuf.shape[2]), dtype=np.float64)
        for i in range(0, imbuf.shape[2]):
            band = imbuf[:, :, i]
            imbuf_mask[:, i] = np.float64(band[mask])               # TODO HACK - check / 5000.  # 5000 is scale for MODIS / XCALIB

        # construct basic per-pixel features

        feat_dict['pan'] = np.mean(imbuf_mask[:, pan_bands], 1)
        # feat_dict['int'] = np.sum(imbuf_mask[:, np.array([2, 3, 5, 6])], 1)     #hack - remove
        if True:
            # constuct features of all possible band/pan ratios
            for key in list(band_dict.keys()):
                feat_dict[key] = imbuf_mask[:, band_dict[key]]
            poly_feat_dict = OrderedDict()
            for key1 in list(feat_dict.keys()):
                poly_feat_dict[key1] = feat_dict[key1]
                for key2 in list(feat_dict.keys()):
                    if not key2 == key1:
                        new_key = '{0}/{1}'.format(key1, key2)
                        poly_feat_dict[new_key] = old_div(feat_dict[key1],feat_dict[key2])
            # poly_feat_dict = feat_dict
        else:
            feat_dict['1/pan'] = 1. / feat_dict['pan']
            for key in list(band_dict.keys()):
                feat_dict[key] = imbuf_mask[:, band_dict[key]]
                feat_dict['1/{0}'.format(key)] = 1./imbuf_mask[:, band_dict[key]]   # invert features to get band ratios with polynomial combinations

            feat_array = np.array(list(feat_dict.values())).transpose()
            poly_feats = PolynomialFeatures(degree=2, interaction_only=False)
            poly_feat_array = np.float64(poly_feats.fit_transform(feat_array))
            poly_feat_names = np.array(poly_feats.get_feature_names())
            my_poly_feat_names = []
            for feat_name in poly_feat_names:       # messy way of re-constructing meaningful feature names
                my_feat_name = ''
                if feat_name == '1':
                    my_feat_name = '1'
                else:
                    ns = np.array(feat_name.split('x'))
                    ns = ns[ns!='']
                    for nss in ns:
                        nsss = np.array(nss.split('^'))
                        my_feat_name += '({0})'.format(list(feat_dict.keys())[int(nsss[0])])
                        if nsss.size > 1:
                            my_feat_name += '^' + nsss[1]
                my_poly_feat_names.append(my_feat_name)
                # print feat_name
                # print my_feat_name

            # remove constant features (x * 1/x)
            const_idx = np.all(np.round(poly_feat_array, 6)==1., axis=0)
            poly_feat_array = poly_feat_array[:, ~const_idx]
            poly_feat_names = poly_feat_names[~const_idx]
            my_poly_feat_names = np.array(my_poly_feat_names)[~const_idx]
            poly_feat_dict = OrderedDict(list(zip(my_poly_feat_names, poly_feat_array.transpose())))

        # add NDVI and SAVI here, to avoid making ratios of them
        SAVI_L = 0.05
        # these vals for MODIS from https://en.wikipedia.org/wiki/Enhanced_vegetation_index
        L = 1.; C1 = 6; C2 = 7.5; G = 2.5
        nir_keys = [key for key in list(band_dict.keys()) if ('NIR' in key) or ('RE' in key)]
        for i, nir_key in enumerate(nir_keys):
            nir = imbuf_mask[:, band_dict[nir_key]]
            r = imbuf_mask[:, band_dict['R']]
            ndvi = old_div((nir - r),(nir + r))
            savi = old_div((1 + SAVI_L) * (nir - r),(SAVI_L + nir + r))
            # evi = G * (nir - r) / (L + nir + C1*r - C2*imbuf_mask[:, band_dict['B']])
            # evi[np.isnan(evi)] = 0; evi[np.isinf(evi)] = 0
            post_fix = '' if nir_key == 'NIR' else '_{0}'.format(nir_key)
            poly_feat_dict['NDVI' + post_fix] = ndvi
            poly_feat_dict['SAVI' + post_fix] = savi
            # poly_feat_dict['EVI' + post_fix] = evi

        if 'NIR2' in band_dict and False:
            nir2 = imbuf_mask[:, band_dict['NIR2']]
            nir = imbuf_mask[:, band_dict['NIR']]
            poly_feat_dict['NDWI'] = old_div((nir2 - nir), (nir2 + nir))

        # poly_feat_dict['NDVI'] = (imbuf_mask[:, band_dict['NIR']] - imbuf_mask[:, band_dict['R']]) / (imbuf_mask[:, band_dict['NIR']] + imbuf_mask[:, band_dict['R']])
        # poly_feat_dict['SAVI'] = (1 + L) * (imbuf_mask[:, band_dict['NIR']] - imbuf_mask[:, band_dict['R']]) / (L + imbuf_mask[:, band_dict['NIR']] + imbuf_mask[:, band_dict['R']])
        # poly_feat_dict['EVI'] = G * (imbuf_mask[:, band_dict['NIR']] - imbuf_mask[:, band_dict['R']]) / \
        #                         (L + imbuf_mask[:, band_dict['NIR']] + C1*imbuf_mask[:, band_dict['R']] - C2*imbuf_mask[:, band_dict['B']])
        if not per_pixel:   # find mean, std dev and ?? features over the plot
            if True:
                plot_feat_dict = OrderedDict()
                for feat_key, feat_vect in poly_feat_dict.items():
                    plot_feat_dict[feat_key] = feat_vect.mean()
                    # note: here some fns are applied to per plot feats and not to per pixel feats (based on theory that only band ratios are necessary/valid per pixel)
                    plot_feat_dict['({0})^2'.format(feat_key)] = feat_vect.mean()**2

                    if include_texture:
                        plot_feat_dict['Std({0})'.format(feat_key)] = feat_vect.std()
                        plot_feat_dict['Entropy({0})'.format(feat_key)] = entropy(feat_vect)

                    if ('VI' in feat_key) or ('ND' in feat_key):        # avoid -ve logs / sqrts
                        plot_feat_dict['1+Log({0})'.format(feat_key)] = np.log10(1. + feat_vect.mean())
                        plot_feat_dict['1+({0})^.5'.format(feat_key)] = np.sqrt(1. + feat_vect.mean())
                    else:
                        plot_feat_dict['Log({0})'.format(feat_key)] = np.log10(feat_vect.mean())
                        plot_feat_dict['({0})^.5'.format(feat_key)] = np.sqrt(feat_vect.mean())
                    # plot_feat_dict['({0})^.5'.format(feat_key)] = np.sqrt(feat_vect.mean())

                    if False: # GLCM features
                        # find pan and quantise to nlevels levels
                        nlevels = 10
                        max_dn = 4000
                        pan_band = np.mean(imbuf[:,:,pan_bands], axis=2)
                        if True:   # global clip normalise
                            pan_band[pan_band < 0] = 0
                            pan_band[pan_band > max_dn] = max_dn
                            pan_band = np.uint8(old_div(((nlevels - 1) * pan_band), max_dn))
                        else:       # local min max normalise
                            pan_band = pan_band - pan_band.min()
                            pan_band = np.uint8(old_div(((nlevels-1)*pan_band),pan_band.max()))
                        cm = greycomatrix(pan_band, distances=[2, 8], angles=[0, old_div(np.pi, 2)], levels=nlevels, symmetric=True, normed=False)
                        for prop_key, prop in {'corr':'correlation', 'diss':'dissimilarity', 'hom':'homogeneity'}.items():
                            d = greycoprops(cm, prop)
                            dd = d.mean(axis=1)
                            plot_feat_dict['{0}(GLCM)'.format(prop_key)] = dd.mean()
                            plot_feat_dict['{0}(GLCM)[0]'.format(prop_key)] = dd[0]
                            plot_feat_dict['{0}(GLCM)[-1]'.format(prop_key)] = dd[-1]
            else:
                plot_feat_dict = OrderedDict()
                for feat_key, feat_vect in poly_feat_dict.items():
                    plot_feat_dict['Mean({0})'.format(feat_key)] = feat_vect.mean()
                    if ('NDVI' in feat_key) or ('SAVI' in feat_key):
                        plot_feat_dict['1+Log({0})'.format(feat_key)] = np.log10(1.+feat_vect.mean())
                    else:
                        plot_feat_dict['Log({0})'.format(feat_key)] = np.log10(feat_vect.mean())
                    plot_feat_dict['Std({0})'.format(feat_key)] = feat_vect.std()
            return plot_feat_dict
        else:
            return poly_feat_dict


    # per_pixel = True, patch_fn = su.ImPlotFeatureExtractor.extract_patch_ms_features
    def extract_all_features(self, per_pixel=False, patch_fn=extract_patch_ms_features_ex):
        # geotransform = ds.GetGeoTransform()
        # transform = osr.CoordinateTransformation(self.plot_feat_dict['spatial_ref'], gdal.osr.SpatialReference(self.image_reader.crs.to_string()))
        # transform = osr.CoordinateTransformation(gdal.osr.SpatialReference(self.plot_data_gdf.crs.to_wkt()),
        #                                          gdal.osr.SpatialReference(self.image_reader.crs.to_wkt()))

        self.plot_data_gdf = self.plot_data_gdf.to_crs(self.image_reader.crs)
        self.plot_data_gdf = self.plot_data_gdf.set_index('ID').sort_index()

        # self.im_data_df = pd.DataFrame()
        im_plot_data_dict = {}

        im_plot_count = 0
        max_thumbnail_vals = np.zeros((self.image_reader.count))

        # loop through plot polygons
        for plot_id, plot in self.plot_data_gdf.iterrows():
            # convert polygon to mask with rio
            plot_mask, plot_transform, plot_window = raster_geometry_mask(self.image_reader, [plot['geometry']], crop=True,
                                                                     all_touched=False)
            plot_cnrs_pixel =  np.fliplr(np.array(plot_window.toranges()).transpose())
            plot_mask = ~plot_mask

            # check plot window lies inside image
            # TODO - is there a rio shortcut for this?
            if not (np.all(plot_cnrs_pixel >= 0) and np.all(plot_cnrs_pixel[:, 0] < self.image_reader.width) \
                    and np.all(plot_cnrs_pixel[:, 1] < self.image_reader.height)):  # and plot.has_key('Yc') and plot['Yc'] > 0.:
                logger.warning(f'Excluding plot {plot["ID"]} - outside image extent')
                continue

            im_buf = self.image_reader.read(window=plot_window)     # read plot ROI from image
            im_buf = np.moveaxis(im_buf, 0, 2)  # TODO get rid of this somehow eg change all imbuf axis orderings to bands first

            if np.all(im_buf == 0) and not patch_fn == self.extract_patch_clf_features:
                logger.warning(f'Excluding plot {plot["ID"]} - all pixels are zero')
                continue

            # amend plot_mask to exclude NAN and -ve pixels
            if False: #patch_fn == self.extract_patch_clf_features:
                plot_mask = plot_mask & np.all(~np.isnan(im_buf), axis=2)
            else:
                plot_mask = plot_mask & np.all(im_buf > 0, axis=2) & np.all(~np.isnan(im_buf), axis=2)

            im_feat_dict = patch_fn(im_buf.copy(), mask=plot_mask, per_pixel=per_pixel)    # extract image features for this plot

            im_data_dict = im_feat_dict.copy()      # copy plot data into feat dict
            for k, v in plot.items():
                im_data_dict[k] = v

            # calculate versions of ABC and AGC with actual polygon area, rather than theoretical plot sizes
            if 'Abc' in plot and 'LitterCHa' in plot:
                litterCHa = np.max([plot['LitterCHa'], 0.])
                abc = np.max([plot['Abc'], 0.])
                im_data_dict['AbcHa2'] = abc * (100. ** 2) /  plot['geometry'].area
                im_data_dict['AgcHa2'] = litterCHa + im_data_dict['AbcHa2']

            # create and store plot thumbnail
            thumbnail = np.float32(im_buf.copy())
            thumbnail[~plot_mask] = 0.
            im_data_dict['thumbnail'] = thumbnail

            im_plot_data_dict[plot_id] = im_data_dict

            # store max thumbnail vals for scaling later
            max_val = np.percentile(thumbnail, 98., axis=(0,1))
            max_thumbnail_vals[max_val > max_thumbnail_vals] = max_val[max_val > max_thumbnail_vals]
            im_plot_count += 1

            log_dict = {'ABC': 'Abc' in plot, 'Num zero pixels': (im_buf == 0).sum(), 'Num -ve pixels': (im_buf < 0).sum(),
                'Num nan pixels': np.isnan(im_buf).sum()}
            logger.info(', '.join([f'Plot {plot_id}'] + ['{}: {}'.format(k, v) for k, v in log_dict.items()]))

        logger.info('Processed {0} plots'.format(im_plot_count))

        # scale thumbnails for display
        for im_data_dict in im_plot_data_dict.values():
            thumbnail = im_data_dict['thumbnail']
            for b in range(0, self.image_reader.count):
                thumbnail[:, :, b] /= max_thumbnail_vals[b]
                thumbnail[:, :, b][thumbnail[:, :, b] > 1.] = 1.
            im_data_dict['thumbnail'] = thumbnail

        # create MultiIndex column labels that separate features from other data
        data_labels = ['feats']*len(im_feat_dict) + ['data']*(len(im_data_dict) - len(im_feat_dict))
        columns = pd.MultiIndex.from_arrays([data_labels, list(im_data_dict.keys())], names=['high','low'])

        self.im_plot_data_gdf = gpd.GeoDataFrame.from_dict(im_plot_data_dict, orient='index')
        self.im_plot_data_gdf.columns = columns
        self.im_plot_data_gdf[('data','ID')] = self.im_plot_data_gdf.index

        return self.im_plot_data_gdf


    # # this version goes with extract_patch_ms_features_ex
    # def get_feat_array_ex(self, y_data_key=None, feat_keys=None):
    #     if self.im_plot_data_gdf is None or len(self.im_plot_data_gdf) == 0:
    #         raise Exception('No features - run extract_all_features() first')
    #
    #     y = np.array([])
    #     if y_data_key is not None:
    #         y = self.im_plot_data_gdf[('data', y_data_key)]
    #
    #     if feat_keys is None:  # get all feats
    #         feat_keys = self.im_plot_data_gdf['feats'].columns
    #         X = self.im_plot_data_gdf['feats']
    #     else:
    #         X = self.im_plot_data_gdf['feats'][feat_keys]
    #
    #
    #     logger.info('X NaN ttl: {0}'.format(np.isnan(X).sum()))
    #     logger.info('X NaN feats: ' + str(feat_keys[np.any(np.isnan(X), axis=0)]))
    #     logger.info('X inf ttl: {0}'.format((X == np.inf).sum()))
    #     logger.info('X inf feats: ' + str(feat_keys[np.any((X == np.inf), axis=0)]))
    #     return X, y, feat_keys


    # @staticmethod
    # def sscatter_plot_(im_feat_dict, x_feat_key='NDVI', y_feat_key='', class_key='', show_labels=True, show_class_labels=True,
    #                  show_thumbnails=False, do_regress=True, xlabel=None, ylabel=None, xfn=lambda x: x, yfn=lambda y: y):
    #     x = np.array([xfn(plot['feats'][x_feat_key]) for plot in list(im_feat_dict.values())])
    #     if type(x[0]) is np.ndarray:        # this is pixel data and requires concat to flatten it
    #         cfn = lambda x: np.hstack(x)[::5]
    #         show_thumbnails = False
    #     else:
    #         cfn = lambda x: x
    #
    #     # if xfn is not None:
    #     #     x = xfn(x)
    #     y = np.array([yfn(plot[y_feat_key]) for plot in list(im_feat_dict.values())])
    #     # if type(x[0]) is np.ndarray:
    #     #     ycfn = lambda x: np.concatenate(x)
    #     # else:
    #     #     ycfn = lambda x: x
    #
    #     # if yfn is not None:
    #     #     y = yfn(y)
    #
    #     if show_class_labels == True:
    #         class_labels = np.array([plot[class_key] for plot in list(im_feat_dict.values())])
    #     else:
    #         class_labels = np.zeros(x.__len__())
    #     if show_thumbnails == True:
    #         thumbnails = np.array([plot['thumbnail'] for plot in list(im_feat_dict.values())])
    #
    #     if show_labels == True:
    #         labels = np.array([plot['ID'] for plot in list(im_feat_dict.values())])
    #
    #     classes = np.unique(class_labels)
    #     colours = ['r', 'm', 'b', 'g', 'y', 'k', 'o']
    #
    #     ylim = [np.min(cfn(y)), np.max(cfn(y))]
    #     xlim = [np.min(cfn(x)), np.max(cfn(x))]
    #     xd = np.diff(xlim)[0]
    #     yd = np.diff(ylim)[0]
    #
    #     pylab.figure()
    #     pylab.axis(np.concatenate([xlim, ylim]))
    #     # pylab.hold('on')
    #     ax = pylab.gca()
    #     handles = np.zeros(classes.size).tolist()
    #     #
    #
    #     for ci, (class_label, colour) in enumerate(zip(classes, colours[:classes.__len__()])):
    #         class_idx = class_labels == class_label
    #         if not show_thumbnails:
    #             pylab.plot(cfn(x[class_idx]), cfn(y[class_idx]), colour + 'o', label=class_label, markeredgecolor=(0, 0, 0))
    #
    #         for xyi, (xx, yy) in enumerate(zip(x[class_idx], y[class_idx])):  # , np.array(plot_names)[class_idx]):
    #             if type(xx) is np.ndarray:
    #                 xx = xx[0]
    #             if type(yy) is np.ndarray:
    #                 yy = yy[0]
    #             if show_labels:
    #                 pylab.text(xx - .0015, yy - .0015, np.array(labels)[class_idx][xyi],
    #                            fontdict={'size': 9, 'color': colour, 'weight': 'bold'})
    #
    #             if show_thumbnails:
    #                 imbuf = np.array(thumbnails)[class_idx][xyi]
    #                 band_idx = [0, 1, 2]
    #                 if imbuf.shape[2] == 8:  # guess wv3
    #                     band_idx = [4, 2, 1]
    #
    #                 ims = 20.
    #                 extent = [xx - old_div(xd, (2 * ims)), xx + old_div(xd, (2 * ims)), yy - old_div(yd, (2 * ims)), yy + old_div(yd, (2 * ims))]
    #                 #pylab.imshow(imbuf[:, :, :3], extent=extent, aspect='auto')  # zorder=-1,
    #                 pylab.imshow(imbuf[:,:,band_idx], extent=extent, aspect='auto')  # zorder=-1,
    #                 handles[ci] = ax.add_patch(
    #                     patches.Rectangle((xx - old_div(xd, (2 * ims)), yy - old_div(yd, (2 * ims))), old_div(xd, ims), old_div(yd, ims), fill=False,
    #                                       edgecolor=colour, linewidth=2.))
    #                 # pylab.plot(mPixels[::step], dRawPixels[::step], color='k', marker='.', linestyle='', markersize=.5)
    #         if do_regress and classes.__len__() > 1 and False:
    #             (slope, intercept, r, p, stde) = stats.linregress(cfn(x[class_idx]), cfn(y[class_idx]))
    #             pylab.text(xlim[0] + xd * 0.7, ylim[0] + yd * 0.05 * (ci + 2),
    #                        '{1}: $R^2$ = {0:.2f}'.format(np.round(r ** 2, 2), classes[ci]),
    #                        fontdict={'size': 10, 'color': colour})
    #
    #     if do_regress:
    #         (slope, intercept, r, p, stde) = stats.linregress(cfn(x), cfn(y))
    #         pylab.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(r ** 2, 2)),
    #                    fontdict={'size': 12})
    #         logger.info('R^2 = {0:.4f}'.format(r ** 2))
    #         logger.info('P (slope=0) = {0:f}'.format(p))
    #         logger.info('Slope = {0:.4f}'.format(slope))
    #         logger.info('Std error of slope = {0:.4f}'.format(stde))
    #         yhat = cfn(x)*slope + intercept
    #         rmse = np.sqrt(np.mean((cfn(y) - yhat) ** 2))
    #         logger.info('RMS error = {0:.4f}'.format(rmse))
    #     else:
    #         r = np.nan
    #         rmse = np.nan
    #
    #     if xlabel is not None:
    #         pylab.xlabel(xlabel, fontdict={'size': 12})
    #     else:
    #         pylab.xlabel(x_feat_key, fontdict={'size': 12})
    #     if ylabel is not None:
    #         pylab.ylabel(ylabel, fontdict={'size': 12})
    #     else:
    #         pylab.ylabel(y_feat_key, fontdict={'size': 12})
    #
    #     pylab.grid()
    #     if classes.size > 0:
    #         if show_thumbnails:
    #             pylab.legend(handles, classes, fontsize=12)
    #         else:
    #             pylab.legend(classes, fontsize=12)
    #     return r**2, rmse
    #
    # @staticmethod
    # def sscatter_plot(im_feat_dict, x_feat_key='NDVI', y_feat_key='', class_key='', show_labels=True, show_class_labels=True,
    #                  show_thumbnails=False, do_regress=True, xlabel=None, ylabel=None, xfn=lambda x: x, yfn=lambda y: y):
    #     x = np.array([xfn(plot['feats'][x_feat_key]) for plot in list(im_feat_dict.values())])
    #     y = np.array([yfn(plot[y_feat_key]) for plot in list(im_feat_dict.values())])
    #
    #     if show_class_labels == True:
    #         class_labels = np.array([plot[class_key] for plot in list(im_feat_dict.values())])
    #     else:
    #         class_labels = None
    #
    #     if show_thumbnails == True:
    #         thumbnails = np.array([plot['thumbnail'] for plot in list(im_feat_dict.values())])
    #     else:
    #         thumbnails = None
    #
    #     if show_labels == True:
    #         labels = np.array([plot['ID'] for plot in list(im_feat_dict.values())])
    #     else:
    #         labels = None
    #
    #     if xlabel is None:
    #         xlabel = x_feat_key
    #     if ylabel is None:
    #         ylabel = y_feat_key
    #     # pylab.ylabel(yf)
    #     return scatter_plot(x, y, class_labels=class_labels, labels=labels, thumbnails=thumbnails,
    #                                                do_regress=do_regress, xlabel=xlabel, ylabel=ylabel, xfn=xfn, yfn=yfn)
    #
    #
    # def scatter_plot(self, x_feat_key='NDVI', y_feat_key='', class_key='', show_labels=True, show_class_labels=True,
    #                  show_thumbnails=False, do_regress=True, xlabel=None, ylabel=None, xfn=lambda x: x, yfn=lambda y: y):
    #     return ImPlotFeatureExtractor.sscatter_plot(self.im_feat_dict, x_feat_key=x_feat_key, y_feat_key=y_feat_key, class_key=class_key, show_labels=show_labels, show_class_labels=show_class_labels,
    #                  show_thumbnails=show_thumbnails, do_regress=do_regress, xlabel=xlabel, ylabel=ylabel, xfn=xfn, yfn=yfn)

class FeatureSelector(object):
    def __init__(self):
        return
    @staticmethod
    def forward_selection(feat_df, y, max_num_feats=0, model=linear_model.LinearRegression(),
                          score_fn=None, cv=None):

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
            for feat_key, feat_vec in available_feats_df.iteritems():
                test_feats_df = pd.concat((selected_feats_df, feat_vec), axis=1, ignore_index=False) # list(selected_feats.values()) + [feat_vec]
                scores, predicted = FeatureSelector.score_model(test_feats_df, y, model=model,
                                                                score_fn=score_fn, cv=cv, find_predicted=False)
                if score_fn is None:
                    score = scores['test_-RMSE'].mean()
                else:
                    score = scores['test_user'].mean()

                if score > best_score:
                    best_score = score
                    best_feat = list(feat_vec)
                    best_key = feat_key
            selected_feats_df[best_key] = best_feat
            selected_scores.append(best_score)
            available_feats_df.pop(best_key)
            logger.info('Feature {0} of {1}: {2}'.format(selected_feats_df.shape[1], max_num_feats, best_key))
        # logger.info(' ')
        selected_scores = np.array(selected_scores)
        selected_feat_keys = selected_feats_df.columns
        best_selected_feat_keys = selected_feat_keys[:np.argmax(selected_scores) + 1]
        logger.info('Best score: {0}'.format(selected_scores.max()))
        logger.info('Num feats at best score: {0}'.format(np.argmax(selected_scores) + 1))
        logger.info('Feat keys at best score: {0}'.format(best_selected_feat_keys))

        return selected_feats_df, selected_scores

    # @staticmethod
    # def forward_selection(X, y, feat_keys=None, max_num_feats=0, model=linear_model.LinearRegression(),
    #                       score_fn=lambda y, pred: -1*np.sqrt(metrics.mean_squared_error(y,pred)), cv=None):
    #     # X, feat_keys_mod, y = self.get_feat_array(y_key=y_feat_key)
    #     if feat_keys is None:
    #         feat_keys = [str(i) for i in range(0, X.shape[1])]
    #     feat_list = X.transpose().tolist()
    #     feat_dict = dict(list(zip(feat_keys, feat_list)))
    #     if max_num_feats == 0:
    #         max_num_feats = X.shape[1]
    #     selected_feats = collections.OrderedDict()   # remember order items are added
    #     selected_scores = []
    #     available_feats = feat_dict
    #
    #     logger.info('Forward selection: ', end=' ')
    #     while len(selected_feats) < max_num_feats:
    #         best_score = -np.inf
    #         best_feat = []
    #         for feat_key, feat_vec in available_feats.items():
    #             test_feats = list(selected_feats.values()) + [feat_vec]
    #             scores, predicted = FeatureSelector.score_model(np.array(test_feats).transpose(), y, model=model, score_fn=score_fn, cv=cv, find_predicted=False)
    #             score = scores['test_user'].mean()
    #             if score > best_score:
    #                 best_score = score
    #                 best_feat = list(feat_vec)
    #                 best_key = feat_key
    #         selected_feats[best_key] = best_feat
    #         selected_scores.append(best_score)
    #         available_feats.pop(best_key)
    #         logger.info(best_key + ', ', end=' ')
    #     logger.info(' ')
    #     selected_scores = np.array(selected_scores)
    #     selected_feat_keys = list(selected_feats.keys())
    #     best_selected_feat_keys = selected_feat_keys[:np.argmax(selected_scores) + 1]
    #     logger.info('Best score: {0}'.format(selected_scores.max()))
    #     logger.info('Num feats at best score: {0}'.format(np.argmax(selected_scores) + 1))
    #     logger.info('Feat keys at best score: {0}'.format(best_selected_feat_keys))
    #
    #     return np.array(list(selected_feats.values())).transpose(), selected_scores, selected_feat_keys

    @staticmethod
    def ranking(X, y, feat_keys=None, model=linear_model.LinearRegression(),
                score_fn=lambda y, pred: -1*np.sqrt(metrics.mean_squared_error(y,pred)), cv=None):
        # X, feat_keys_mod, y = self.get_feat_array(y_key=y_feat_key)
        if feat_keys is None:
            feat_keys = [str(i) for i in range(0, X.shape[1])]
        feat_list = X.transpose().tolist()
        feat_dict = OrderedDict(list(zip(feat_keys, feat_list)))
        feat_scores = []
        for i, (feat_key, feat_vect) in enumerate(feat_dict.items()):
            if False:
                score = score_fn(y, np.array(feat_vect).reshape(-1, 1))
            else:
                scores, predicted = FeatureSelector.score_model(np.array(feat_vect).reshape(-1, 1), y.reshape(-1, 1), model=model,
                                                           score_fn=score_fn, cv=cv, find_predicted=True)
                score = scores['test_user'].mean()
                # score = scores['R2_stacked']        #hack
            feat_scores.append(score)
            if i%20==0:
                logger.info('.', end=' ')
        logger.info(' ')
        feat_scores = np.array(feat_scores)

        logger.info('Best score: {0}'.format(feat_scores.max()))
        logger.info('Best feat: {0}'.format(feat_keys[np.argmax(feat_scores)]))
        return feat_scores

    @staticmethod
    def score_model(X, y, model=linear_model.LinearRegression(), score_fn=None,
                    cv=None, find_predicted=True, print_scores=False):

        # X = np.array(feat_list).transpose()
        # y = np.array([plot[y_feat_key] for plot in self.im_feat_dict.values()])
        # y = np.hstack([np.tile(plot[y_feat_key], plot[feat_keys[0]].size) for plot in self.im_feat_dict.values()])
        predicted = None
        if cv is None:
            cv = len(y)        # Leave one out

        if score_fn is not None:
            scoring = {#'R2': make_scorer(metrics.r2_score),        # R2 in cross validation is suspect
                       '-RMSE': make_scorer(lambda y, pred: -np.sqrt(metrics.mean_squared_error(y, pred))),
                       'user': make_scorer(score_fn)}
        else:
            scoring = {'-RMSE': make_scorer(lambda y, pred: -np.sqrt(metrics.mean_squared_error(y, pred)))}

        scores = cross_validate(model, X, y, scoring=scoring, cv=cv)

        if print_scores:
            rmse_ci = np.percentile(-scores['test_-RMSE'], [5, 95])
            # r2_ci = np.percentile(-scores['test_R2'], [5, 95])
            logger.info('RMSE mean: {0:.4f}, std: {1:.4f}, 5-95%: {2:.4f} - {3:.4f}'.format(-scores['test_-RMSE'].mean(),
                    scores['test_-RMSE'].std(), rmse_ci[0], rmse_ci[1]))
            # logger.info('R2 mean: {0:.4f}, std: {1:.4f}, 5-95%: {2:.4f} - {3:.4f}'.format(scores['test_R2'].mean(), scores['test_R2'].std(),
            #                                                                               r2_ci[0], r2_ci[1]))
        if find_predicted:
            predicted = cross_val_predict(model, X, y, cv=cv)  #)
            scores['R2_stacked'] = metrics.r2_score(y, predicted)   # Also suspect for validation, but better than cross validated R2
            if print_scores:
                logger.info('R2 (stacked): {0:.4f}'.format(scores['R2_stacked']))
        return scores, predicted
        # score = {}
        # score = score_fn(y, predicted)
        # score['r2'] = metrics.r2_score(y, predicted)
        # score['rmse'] = np.sqrt(metrics.mean_squared_error(y, predicted))
        # score['mse'] = (-scores['test_neg_mean_squared_error'])
        # score['rms'] = np.sqrt(-scores['test_neg_mean_squared_error'])


    # assumes X is 2D and scaled 0-1, clf is a trained classifier


# params: calib_plots from >1 data set
#       model_data_plots from >1 data set
#       (for now the above are the same thing)
#       1 data_set is specified as fitted one, the rest are tests, this can also be done sort of cross-validated
#       a model spec i.e. feature indices and model type
#       num calib plots to use


class ApplyLinearModel(object):
    def __init__(self, in_file_name='', out_file_name='', model=linear_model.LinearRegression, model_keys=[],
                 feat_ex_fn=ImPlotFeatureExtractor.extract_patch_ms_features_ex, num_bands=9, save_feats=False):
        self.in_file_name = in_file_name
        self.out_file_name = out_file_name
        self.model = model
        self.model_keys = model_keys
        self.feat_ex_fn = feat_ex_fn
        self.pan_bands = []
        self.band_dict = {}
        self.save_feats = save_feats
        self.nodata = np.nan

        input_ds = None
        output_ds = None

    # def __del__(self):
    #     self.Close()

    # Contruct fn ptrs for each feature from strings
    # (so we don't have to find the entire feature library)
    # Examples:
    #     'Log(R/pan)',
    #     'Std(pan/NIR2)',
    #     'Log(B)',
    #     'Entropy(NIR/Y)',
    #     '(NIR/Y)^2',
    # def ConstructFeatEx(self, model_keys=[], num_bands=9):
    @staticmethod
    def construct_feat_ex_fn(model_key='', pan_bands=None, band_dict=None):
        win_fn_list = []
        inner_str_list = []
        inner_fn_list = []
        # find outer fn
        ks = re.split('\(|\)', model_key.lower())
        if len(ks) == 1:    # no brackets
            inner_str = ks[0]
            win_fn = lambda x: np.nanmean(x, axis=(0, 2))
        else:
            inner_str = ks[1]
            # outer fn
            if not ks[0]:  # nothing before (
                if ks[-1]:  # raise to power
                    kp = re.split('\^', ks[-1])
                    win_fn = lambda x: np.power(np.nanmean(x, axis=(0,2)), eval(kp[1]))
                else:  # straight mean
                    win_fn = lambda x: np.nanmean(x, axis=(0,2))
            elif ks[0] == 'std':
                win_fn = lambda x: np.nanstd(x, axis=(0,2))
            elif ks[0] == 'entropy':
                win_fn = lambda x: nanentropy(x, axis=(0,2)).astype('float32')
            elif ks[0] == 'log':
                win_fn = lambda x: np.log10(np.nanmean(x, axis=(0,2)))

        # find inner fn i.e. band ratio
        iss = re.split('/', inner_str)
        if len(iss) > 1:
            if iss[0] == 'pan':
                inner_fn = lambda pan, x: pan / x[band_dict[iss[1].upper()], :, :]
            elif iss[1] == 'pan':
                inner_fn = lambda pan, x: x[band_dict[iss[0].upper()], :, :] / pan
            else:
                inner_fn = lambda pan, x: x[band_dict[iss[0].upper()], :, :] / x[band_dict[iss[1].upper()], :, :]
        elif 'savi' in inner_str:   # TO DO: add in underscore options
            if 'nir2' in inner_str:
                inner_fn = lambda pan, x: (1 + 0.05) * (x[band_dict['NIR2'], :, :] - x[band_dict['R'], :, :]) / (
                            0.05 + x[band_dict['NIR2'], :, :] + x[band_dict['R'], :, :])
            elif 're' in inner_str:
                inner_fn = lambda pan, x: (1 + 0.05) * (x[band_dict['RE'], :, :] - x[band_dict['R'], :, :]) / (0.05 + x[band_dict['RE'], :, :] + x[band_dict['R'], :, :])
            else:
                inner_fn = lambda pan, x: (1 + 0.05) * (x[band_dict['NIR'], :, :] - x[band_dict['R'], :, :]) / (0.05 + x[band_dict['NIR'], :, :] + x[band_dict['R'], :, :])

        elif 'ndvi' in inner_str:
            if 'nir2' in inner_str:
                inner_fn = lambda pan, x: (x[band_dict['NIR2'], :, :] - x[band_dict['R'], :, :]) / (x[band_dict['NIR2'], :, :] + x[band_dict['R'], :, :])
            elif 're' in inner_str:
                inner_fn = lambda pan, x: (x[band_dict['RE'], :, :] - x[band_dict['R'], :, :]) / (x[band_dict['RE'], :, :] + x[band_dict['R'], :, :])
            else:
                inner_fn = lambda pan, x: (x[band_dict['NIR'], :, :] - x[band_dict['R'], :, :]) / (x[band_dict['NIR'], :, :] + x[band_dict['R'], :, :])
        else:
            if iss[0] == 'pan':
                inner_fn = lambda pan, x: pan
            else:
                inner_fn = lambda pan, x: x[band_dict[iss[0].upper()], :, :]

        return inner_str, win_fn, inner_fn

    @staticmethod
    def construct_feat_ex_fns(model_keys=[], num_bands=9):
        import re
        pan_bands, band_dict = mdl.ImPlotFeatureExtractor.get_band_info(num_bands)

        win_fn_list = []
        inner_str_list = []
        inner_fn_list = []
        for model_key in model_keys:
            # find outer fn
            inner_str, win_fn, inner_fn = ApplyLinearModel.construct_feat_ex_fn(model_key, pan_bands=pan_bands, band_dict=band_dict)
            win_fn_list.append(win_fn)
            inner_str_list.append(inner_str)
            inner_fn_list.append(inner_fn)
        # inner fn
        return win_fn_list, inner_fn_list

    # @staticmethod
    # def rolling_window(a, window, step_size=1):
    #     shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    #     strides = a.strides + (a.strides[-1] * step_size,)
    #     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

    @staticmethod
    def rolling_window(a, window, step_size=1):
        shape = a.shape[:-1] + (int(1 + (a.shape[-1] - window) / step_size), window)
        strides = a.strides[:-1] + (step_size * a.strides[-1], a.strides[-1])
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

    def feat_ex(self, im_buf=[]):
        return

    def create(self, win_size=(1, 1), step_size=(1, 1)):
        from rasterio.windows import Window
        with rasterio.Env():
            with rasterio.open(self.in_file_name, 'r') as in_ds:
                if self.save_feats:
                    out_bands = len(self.model_keys) + 1
                else:
                    out_bands = 1
                out_profile = in_ds.profile
                out_size = np.floor([1 + (in_ds.width - win_size[0])/step_size[0], 1 + (in_ds.height - win_size[1])/step_size[1]]).astype('int32')
                out_profile.update(dtype=rasterio.float32, count=out_bands, compress='deflate', driver='GTiff', width=out_size[0], height=out_size[1], nodata=self.nodata)

                out_profile['transform'] = out_profile['transform']*rasterio.Affine.scale(step_size[0])
                if (out_size[0]/out_profile['blockxsize'] < 10) | (out_size[1]/out_profile['blockysize'] < 10):
                    out_profile['tiled'] = False
                    out_profile['blockxsize'] = out_size[0]
                    out_profile['blockysize'] = 1
                # out_transform = np.array(rasterio.Affine(out_profile['transform']).to_gdal())
                # out_transform[0] = ['transform'][4] * step_size[0]
                # out_transform[4] = ['transform'][4] * step_size[1]
                # to do: find geotransform based on out width/height

                # self.win_fn_list, self.band_ratio_list = AgcMap.construct_feat_ex_fns(self.model_keys, num_bands=in_ds.count)

                with rasterio.open(self.out_file_name, 'w', **out_profile) as out_ds:
                    pan_bands, band_dict = ImPlotFeatureExtractor.get_band_info(in_ds.count)
                    win_off = np.floor(np.array(win_size)/(2*step_size[0])).astype('int32')
                    prog_update = 10
                    for cy in range(0, in_ds.height - win_size[1] + 1, step_size[1]):     #12031): #
                        # read row of windows into mem and slide the win on this rather than the actual file
                        # NB rasterio index is x,y, numpy index is row, col (i.e. y,x)

                        # in_buf = np.zeros((win_size[1], in_ds.width), dtype=in_ds.dtypes[0])
                        in_win = Window(0, cy, out_size[0]*win_size[0], win_size[1])
                        bands = list(range(1, in_ds.count + 1))
                        in_buf = in_ds.read(bands, window=in_win).astype(rasterio.float32)  # NB bands along first dim

                        # TO DO:  deal with -ve vals i.e. make all nans, mask out or something
                        pan = in_buf[pan_bands, :, :].mean(axis=0)
                        in_nan_mask = np.any(in_buf <= 0, axis=0) | np.any(pan == 0, axis=0)  # this is overly conservative but neater/faster
                        # agc_buf = self.model.intercept_ * np.ones((1, in_ds.width - win_size[0] + 1), dtype=out_ds.dtypes[0])
                        agc_buf = self.model.intercept_ * np.ones((out_size[0]), dtype=out_ds.dtypes[0])
                        out_win = Window(win_off[0], int(cy/step_size[0]) + win_off[1], out_size[0], 1)
                        # for i, (win_fn, band_ratio_fn) in enumerate(zip(self.win_fn_list, self.band_ratio_list)):
                        for i, model_key in enumerate(self.model_keys):
                            inner_str, win_fn, band_ratio_fn = ApplyLinearModel.construct_feat_ex_fn(model_key, pan_bands=pan_bands,
                                                                                                     band_dict=band_dict)
                            band_ratio = band_ratio_fn(pan, in_buf)
                            band_ratio[in_nan_mask] = np.nan           # to exclude from window stats/fns
                            feat_buf = win_fn(ApplyLinearModel.rolling_window(band_ratio, win_size[0], step_size=step_size[0])) * self.model.coef_[i]
                            agc_buf += feat_buf
                            if i==0:
                                first_feat_buf = feat_buf
                            if self.save_feats:
                                if feat_buf.size == 1:
                                    feat_buf = np.zeros((1, out_size[0]), dtype=out_ds.dtypes[0])
                                else:
                                    feat_buf[np.isinf(feat_buf) | np.isnan(feat_buf)] = 0
                                out_ds.write(feat_buf.reshape(1,-1), indexes=2+i, window=out_win)

                        if True:    # lower limit of 0, and set noisy/invalid pixels to nodata, so that they can be filled in a post-processing step
                            nodata_mask = np.isinf(agc_buf) | np.isnan(agc_buf) # | (agc_buf > 70)        # 55/70 comes from looking ta the histogram
                            agc_buf[nodata_mask] = self.nodata
                            # agc_buf = sig.medfilt(agc_buf, kernel_size=3).astype(np.float32)
                            # agc_buf[(agc_buf < 0)] = 0
                        else:   # inelegant hack for suspect values
                            nodata_mask = np.isinf(agc_buf) | np.isnan(agc_buf) | (agc_buf < 0) | (agc_buf > 100)
                            agc_buf[nodata_mask] = first_feat_buf[nodata_mask]
                        # out_win = Window(win_off[0], cy + win_off[1], in_ds.width - win_size[0] + 1, 1)
                        out_ds.write(agc_buf.reshape(1,-1), indexes=1, window=out_win)

                        if np.floor(old_div(100*cy, in_ds.height)) > prog_update:
                            print('.', end=' ')
                            prog_update += 10
                            # break
                    print(' ')

    def post_proc(self):
        from rasterio.windows import Window
        from rasterio import fill
        with rasterio.Env():
            with rasterio.open(self.out_file_name, 'r') as in_ds:
                out_profile = in_ds.profile
                out_profile.update(count=1)
                split_ext = os.path.splitext(self.out_file_name)
                out_file_name = '{0}_postproc{1}'.format(split_ext[0], split_ext[1])
                with rasterio.open(out_file_name, 'w', **out_profile) as out_ds:

                    if (not out_profile['tiled']) or (np.prod(in_ds.shape) < 10e6):
                        in_windows = enumerate([Window(0,0,in_ds.width, in_ds.height)])        # read whole raster at once
                    else:
                        in_windows = in_ds.block_windows(1)                         # read in blocks
                    # in_masks = in_ds.read_masks()
                    # sieved_msk = sieve(in_mask, size=500)
                    for ji, block_win in in_windows:
                        in_block = in_ds.read(1, window=block_win, masked=True)

                        in_block[in_block < 0] = 0
                        in_block.mask = (in_block.mask.astype(np.bool) | (in_block > 95) | (in_block < 0)).astype(rasterio.uint8)

                        in_mask = in_block.mask.copy()
                        sieved_msk = sieve(in_mask.astype(rasterio.uint8), size=2000)

                        out_block = fill.fillnodata(in_block, mask=None, max_search_distance=20, smoothing_iterations=1)
                        out_block[sieved_msk.astype(np.bool)] = self.nodata
                        # out_block = signal.medfilt2d(in_block, kernel_size=3)
                        out_ds.write(out_block, indexes=1, window=block_win)
                        # out_ds.write_mask(~sieved_msk, window=block_win)

                # with rasterio.open(out_file_name, 'r+', **out_profile) as out_ds:
                #     tmp_masks = out_ds.read_masks()