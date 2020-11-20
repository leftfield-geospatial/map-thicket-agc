"""
  GEF5-SLM: Above ground carbon estimation in thicket using multi-spectral images
  Copyright (C) 2020 Dugal Harris
  Released under GNU Affero General Public License (AGPL) (https://www.gnu.org/licenses/agpl.html)
  Email: dugalh@gmail.com
"""

import sys, warnings, logging, os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import patches
from scipy import stats as stats
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.preprocessing import PolynomialFeatures
import rasterio
from rasterio.features import sieve
from rasterio.windows import Window
from rasterio.mask import raster_geometry_mask
import geopandas as gpd, pandas as pd

if sys.version_info.major == 3:
    from sklearn.metrics import make_scorer
else:
    from sklearn.metrics.scorer import make_scorer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def nanentropy(x, axis=None):
    """
    Find entropy ignoring NaNs

    Parameters
    ----------
    x : numpy.array_like
        data in 3 or fewer dimensions
    axis : int
        specify dimension(s) of x along which to find entropy, default=None in which case all dimensions are used

    Returns
    -------
    Entropy of x long specified dimensions
    """
    if not axis is None:
        if (len(axis) > x.ndim) or (np.any(np.array(axis) > x.ndim)):
            raise Exception('len(axis) > x.ndim) or (np.any(axis > x.ndim))')
        elif len(axis) == x.ndim:
            axis = None

    if axis is None:    # find entropy along all dimensions of x
        nbins = 100
        # quantise x into nbins bins
        x = x[~np.isnan(x)]
        if len(x) < 10:
            return 0.
        x = x - x.min()
        x = np.int32(np.round(np.float32(nbins * x) / x.max()))

        value, counts = np.unique(x, return_counts=True)
        p = np.array([count/float(x.size) for count in counts])
        return -np.sum(p * np.log2(p))
    else:        # slice 3D array
        along_axis = np.setdiff1d(range(0, x.ndim), axis)[0]
        e = np.zeros((x.shape[along_axis]))
        for slice in range(x.shape[along_axis]):
            if along_axis == 0:
                xa = x[slice,:,:]
            elif along_axis == 1:
                xa = x[:,slice,:]
            elif along_axis == 2:
                xa = x[:,:,slice]
            else:
                raise Exception('along_axis > 2 or < 0')
            e[slice] = nanentropy(xa, axis=None)
        return e

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

    xlim = [x.min(), x.max()]
    ylim = [y.min(), y.max()]
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
        scores, predicted = FeatureSelector.score_model(x.to_numpy().reshape(-1,1), y.to_numpy().reshape(-1,1), model=linear_model.LinearRegression(),
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
        pyplot.xlabel(x_col[-1], fontdict={'size': 12})

    if y_label is not None:
        pyplot.ylabel(y_label, fontdict={'size': 12})
    else:
        pyplot.ylabel(y_col[-1], fontdict={'size': 12})

    if n_classes > 1:
        if not thumbnail_col is None:
            pyplot.legend(handles, classes, fontsize=12)
        else:
            pyplot.legend(classes, fontsize=12)
    pyplot.show()
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
        scores dict as returned by FeatureSelector.score_model()
    xlabel : str
        label for x axis (optional)
    ylabel : str
        label for y axis (optional)
    """

    df = pd.DataFrame({xlabel: y, ylabel: pred})    # form a datafram for scatter_ds
    scatter_ds(df, do_regress=False)

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
    pyplot.pause(0.1)   # hack to get around pyplot bug when saving figure


class PatchFeatureExtractor():
    def __init__(self, apply_rolling_window=False, rolling_window_xsize=None, rolling_window_xstep=None):
        """
        Virtual class for features from image patches.
        Optional application of rolling window (x direction only)

        Parameters
        ----------
        apply_rolling_window : bool
            include rolling window in feature functions
        rolling_window_xsize : int
            x size of rolling window in pixels (optional)
        rolling_window_xstep : int
            Number of pixels to step in x direction (optional)
        """
        self.fn_dict = {}
        if apply_rolling_window==True:
            if rolling_window_xsize is None or rolling_window_xstep is None:
                raise Exception("rolling_window_xsize and rolling_window_xstep must be specified")
        self._apply_rolling_window = apply_rolling_window
        self._rolling_window_xsize = rolling_window_xsize
        self._rolling_window_xstep = rolling_window_xstep
        self.generate_fn_dict()
        return

    def _rolling_window_view(self, x):
        """
        Return a 3D strided view of 2D array to allow fast rolling window operations.
        Rolling windows are stacked along the third dimension.  No data copying is involved.

        Parameters
        ----------
        x : numpy.array_like
            array to return view of

        Returns
        -------
        3D rolling window view of x
        """

        shape = x.shape[:-1] + (self._rolling_window_xsize, int(1 + (x.shape[-1] - self._rolling_window_xsize) / self._rolling_window_xstep))
        strides = x.strides[:-1] + (x.strides[-1], self._rolling_window_xstep * x.strides[-1])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)

    def generate_fn_dict(self):
        """
        Generate feature extraction dictionary self.fn_dict with values= pointer to feature extraction functions, and
        keys= descriptive feature strings.  Suitable for use with modelling from patches or mapping a whole image i.e.
        ImageFeatureExtractor and ApplyLinearModel.  Generates band ratios, veg. indices, simple texture measures, and
        non-linear transformations thereof.

        Parameters
        ----------
        apply_rolling_window : bool
            Apply window function inside a rolling window

        """
        raise NotImplementedError()

    def extract_features(self, im_patch, mask=None):
        """
        Virtual method to extract features from image patch with optional mask
        """
        raise NotImplementedError()


class MsPatchFeatureExtractor(PatchFeatureExtractor):
    # TODO: PatchFeatureExtractor and MsPatchFeatureExtractor could use some refactoring that gets around the
    #  necessity for apply_rolling_window which is confusing e.g. rolling_window_xsize and rolling_window_xstep are
    #  always passed to extract_features(), and not to __init__. extract_features() then passes to feature functions in
    #  fn_dict.  If there is no rolling window then rolling_window_xsize and rolling_window_xstep are set to the patch x
    #  size.  This would avoid the need for upfront specification of rolling window.

    def __init__(self, num_bands=9, apply_rolling_window=False, rolling_window_xsize=None, rolling_window_xstep=None):
        """
        PatchFeatureExtractor for extracting band ratio, vegetation index and texture features from multi-spectral image patches.

        Parameters
        ----------
        num_bands : int
            number of multi-spectral bands
        apply_rolling_window : bool
            include rolling window in feature functions
        rolling_window_xsize : int
            x size of rolling window in pixels (optional).  If None, no rolling window applied
        rolling_window_xstep : int
            Number of pixels to step in x direction (optional).
        """
        self._num_bands = num_bands
        self._pan_bands, self._band_dict = self.get_band_info(num_bands)
        PatchFeatureExtractor.__init__(self, apply_rolling_window=apply_rolling_window, rolling_window_xsize=rolling_window_xsize,
                                       rolling_window_xstep=rolling_window_xstep)

    @staticmethod
    def get_band_info(num_bands=9):
        """
        Get an array of band numbers to sum to generate panchromatic, and a dict of band labels and numbers for
        assumed multi-spectral images (Worldview3 or NGI aerial)

        Parameters
        ----------
        num_bands : int
            number of multi-spectral bands in the image

        Returns
        -------
        (pan_bands, band_dict), where
            pan_bands: array of band numbers to sum to create pan
            band_dict: dictions of {band name: band number} for multi-spectral bands
        """
        if num_bands == 8:  # assume WV3
            # pan_bands = np.array([1, 2, 3, 4, 5, 6])
            pan_bands = [1, 2, 4, 5]
            band_dict = OrderedDict([('C', 0), ('B', 1), ('G', 2), ('Y', 3), ('R', 4), ('RE', 5), ('NIR', 6), ('NIR2', 7)])
        elif num_bands == 9:  # assume WV3 MS & PAN in band 0
            # pan_bands = np.array([0])     # old version
            pan_bands = [2, 3, 5, 6]
            band_dict = OrderedDict(
                [('C', 1), ('B', 2), ('G', 3), ('Y', 4), ('R', 5), ('RE', 6), ('NIR', 7), ('NIR2', 8)])
        else:  # assume NGI aerial
            pan_bands = [0, 1, 2, 3]
            band_dict = OrderedDict([('R', 0), ('G', 1), ('B', 2), ('NIR', 3)])
        return pan_bands, band_dict

    def generate_fn_dict(self):
        """
        Generate feature extraction dictionary with values= pointers to feature extraction functions, and
        keys= descriptive feature strings.
        Suitable for use with modelling from patches or mapping a whole image i.e. ImageFeatureExtractor and ApplyLinearModel.
        Generates band ratios, veg. indices, simple texture measures, and non-linear transformations thereof.
        """

        if len(self.fn_dict) > 0:
            return

        # inner band ratios
        self.inner_dict = OrderedDict()
        self.inner_dict['pan/1'] = lambda pan, bands: pan
        for num_key in list(self._band_dict.keys()):
            self.inner_dict['{0}/pan'.format(num_key)] = lambda pan, bands, num_key=num_key: bands[self._band_dict[num_key], :] / pan
            self.inner_dict['pan/{0}'.format(num_key)] = lambda pan, bands, num_key=num_key: pan / bands[self._band_dict[num_key], :]
            self.inner_dict['{0}/1'.format(num_key)] = lambda pan, bands, num_key=num_key: bands[self._band_dict[num_key], :]
            for den_key in list(self._band_dict.keys()):
                if not num_key == den_key:
                    self.inner_dict['{0}/{1}'.format(num_key, den_key)] = lambda pan, bands, num_key=num_key, den_key=den_key: \
                        bands[self._band_dict[num_key], :] / bands[self._band_dict[den_key], :]

        # inner veg indices
        SAVI_L = 0.05   # wikipedia
        nir_keys = [key for key in list(self._band_dict.keys()) if ('NIR' in key) or ('RE' in key)]
        for nir_key in nir_keys:
            post_fix = '' if nir_key == 'NIR' else '_{0}'.format(nir_key)
            self.inner_dict['NDVI' + post_fix] = lambda pan, bands, nir_key=nir_key: 1 + (bands[self._band_dict[nir_key], :] - bands[self._band_dict['R'], :]) / \
                                                                                     (bands[self._band_dict[nir_key], :] + bands[self._band_dict['R'], :])
            self.inner_dict['SAVI' + post_fix] = lambda pan, bands, nir_key=nir_key: 1 + (1 + SAVI_L) * (bands[self._band_dict[nir_key], :] - bands[self._band_dict['R'], :]) / \
                                                                                     (SAVI_L + bands[self._band_dict[nir_key], :] + bands[self._band_dict['R'], :])

        # window functions
        if self._apply_rolling_window:
            self.win_dict = OrderedDict({'mean': lambda x: np.nanmean(self._rolling_window_view(x), axis=(0, 1)),
                                         'std': lambda x: np.nanstd(self._rolling_window_view(x), axis=(0, 1)),
                                         'entropy': lambda x: nanentropy(self._rolling_window_view(x), axis=(0, 1))})
        else:
            self.win_dict = OrderedDict({'mean': lambda x: np.nanmean(x, axis=(0,1)), 'std': lambda x: np.nanstd(x, axis=(0,1)), 'entropy': lambda x: nanentropy(x, axis=(0,1))})

        self.scale_dict = OrderedDict({'log': np.log10, 'sqr': lambda x: np.power(x, 2), 'sqrt': np.sqrt})

        # combine inner, window and scaling functions
        for inner_key, inner_fn in self.inner_dict.items():
            for win_key, win_fn in self.win_dict.items():
                fn_key = '({0}({1}))'.format(win_key, inner_key)
                fn = lambda pan, bands, win_fn=win_fn, inner_fn=inner_fn: win_fn(inner_fn(pan, bands))
                self.fn_dict[fn_key] = fn
                if win_key == 'mean':   # for backward compatibility - TODO: apply scaling to all windows
                    for scale_key, scale_fn in self.scale_dict.items():
                        fn_key = '{0}({1}({2}))'.format(scale_key, win_key, inner_key)
                        fn = lambda pan, bands, scale_fn=scale_fn, win_fn=win_fn, inner_fn=inner_fn: scale_fn(win_fn(inner_fn(pan, bands)))
                        self.fn_dict[fn_key] = fn


    def extract_features(self, im_patch, mask=None, fn_keys=None):
        """
        Extract dictionary of features for multi-spectral image patch with optional mask

        Parameters
        ----------
        im_patch : numpy.array_like
            multi-spectal image patch - bands along first dimension (as with rasterio)
        mask : numpy.array_like
            mask of pixels to include (optional) - same x-y size as im_patch
        fn_keys : list
            feature keys to extract (optional - default extract all)

        Returns
        -------
        Dictionary of features feat_dict = {'<feature string>': <feature value>, ...}
        """
        if fn_keys is None:
            fn_keys = self.fn_dict.keys()

        if mask is None:
            mask = np.all(im_patch>0, axis=0)

        mask = np.bool8(mask)
        im_patch_mask = np.float64(im_patch)
        im_patch_mask[:, ~mask] = np.nan
        if self._num_bands != im_patch.shape[0]:
            raise Exception("im_patch must have the same number of bands as passed to MsPatchFeatureExtractor(...)")
        pan_mask = im_patch_mask[self._pan_bands, :].mean(axis=0)      # TODO: fix for len(pan_bands)=1

        feat_dict = OrderedDict()
        for fn_key in fn_keys:
            feat_dict[fn_key] = self.fn_dict[fn_key](pan_mask, im_patch_mask)

        return feat_dict


class MsImageFeatureExtractor(object):
    # TODO: this doesn't need to be a class - refactor as function
    def __init__(self, image_filename=None, plot_data_gdf=gpd.GeoDataFrame()):
        """
        Class to extract features from patches (e.g. ground truth plots) in an image

        Parameters
        ----------
        image_filename : str
            path to image file from which to extract features
        plot_data_gdf : geopandas.GeoDataFrame
            plot polygons with optional ground truth and index of plot ID strings
        """
        self.im_plot_data_gdf = gpd.GeoDataFrame()  # geodataframe of features in subindex 'feats' and data in 'data'
        self._image_reader = rasterio.open(image_filename, 'r')
        self._plot_data_gdf = plot_data_gdf
        self._patch_feature_extractor = MsPatchFeatureExtractor(num_bands=self._image_reader.count)

    def __del__(self):
        self._image_reader.close()

    def extract_image_features(self):
        """
        Extract features from plot polygons specified by self.plot_data_gdf in image

        Returns
        -------
        geopandas.GeoDataFrame of features in subindex 'feats' and data in 'data'
        """

        self._plot_data_gdf = self._plot_data_gdf.to_crs(self._image_reader.crs)   # convert plot co-ordinates to image projection

        im_plot_data_dict = {}      # dict to store plot features etc
        im_plot_count = 0
        max_thumbnail_vals = np.zeros((self._image_reader.count))    # max vals for each image band to scale thumbnails

        for plot_id, plot in self._plot_data_gdf.iterrows():     # loop through plot polygons
            # convert polygon to raster mask
            plot_mask, plot_transform, plot_window = raster_geometry_mask(self._image_reader, [plot['geometry']], crop=True,
                                                                          all_touched=False)
            plot_cnrs_pixel =  np.array(plot_window.toranges())
            plot_mask = ~plot_mask  # TODO: can we lose this?

            # check plot window lies inside image
            if not (np.all(plot_cnrs_pixel[1, :] < self._image_reader.width) and np.all(plot_cnrs_pixel[0, :] < self._image_reader.height) \
                    and np.all(plot_cnrs_pixel >= 0)):  # and plot.has_key('Yc') and plot['Yc'] > 0.:
                logger.warning(f'Excluding plot {plot["ID"]} - outside image extent')
                continue

            im_buf = self._image_reader.read(window=plot_window)     # read plot ROI from image

            if np.all(im_buf == 0):
                logger.warning(f'Excluding plot {plot["ID"]} - all pixels are zero')
                continue

            plot_mask = plot_mask & np.all(im_buf > 0, axis=0) & np.all(~np.isnan(im_buf), axis=0)  # exclude any nan or -ve pixels

            # create plot thumbnail for visualisation
            thumbnail = np.float32(np.moveaxis(im_buf, 0, 2))
            thumbnail[~plot_mask] = 0.

            im_feat_dict = self._patch_feature_extractor.extract_features(im_buf, mask=plot_mask)  # extract image features for this plot

            im_data_dict = {**im_feat_dict, **plot}     # combine features and other plot data
            im_data_dict['thumbnail'] = thumbnail

            # calculate versions of ABC and AGC normalised by actual polygon area, rather than theoretical plot sizes
            if 'Abc' in plot and 'LitterCHa' in plot:
                litterCHa = np.max([plot['LitterCHa'], 0.])
                abc = np.max([plot['Abc'], 0.])
                im_data_dict['AbcHa2'] = abc * (100. ** 2) /  plot['geometry'].area
                im_data_dict['AgcHa2'] = litterCHa + im_data_dict['AbcHa2']

            im_plot_data_dict[plot_id] = im_data_dict       # add to dict of all plots

            # calc max thumbnail vals for scaling later
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
            for b in range(0, self._image_reader.count):
                thumbnail[:, :, b] /= max_thumbnail_vals[b]
                thumbnail[:, :, b][thumbnail[:, :, b] > 1.] = 1.
            im_data_dict['thumbnail'] = thumbnail

        # create MultiIndex column labels that separate features from other data
        data_labels = ['feats']*len(im_feat_dict) + ['data']*(len(im_data_dict) - len(im_feat_dict))
        columns = pd.MultiIndex.from_arrays([data_labels, list(im_data_dict.keys())], names=['high','low'])

        # create geodataframe of results
        self.im_plot_data_gdf = gpd.GeoDataFrame.from_dict(im_plot_data_dict, orient='index')
        self.im_plot_data_gdf.columns = columns
        self.im_plot_data_gdf[('data','ID')] = self.im_plot_data_gdf.index

        return self.im_plot_data_gdf


class FeatureSelector(object):
    # TODO: again, this does not need to be a class, perhaps separate files (agc_estimation) makes better sense
    def __init__(self):
        return

    @staticmethod
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
            for feat_key, feat_vec in available_feats_df.iteritems():
                test_feats_df = pd.concat((selected_feats_df, feat_vec), axis=1, ignore_index=False) # list(selected_feats.values()) + [feat_vec]
                scores, predicted = FeatureSelector.score_model(test_feats_df, y, model=model,
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


    @staticmethod
    def ranking(feat_df, y, model=linear_model.LinearRegression(), score_fn=None, cv=None):

        logger.info('Ranking: ')
        if score_fn is None:
            logger.info('Using negative RMSE score')
        else:
            logger.info('Using user score')

        feat_scores = []
        for i, (feat_key, feat_vec) in enumerate(feat_df.iteritems()):
            logger.info('Scoring feature {0} of {1}'.format(i+1, feat_df.shape[1]))

            scores, predicted = FeatureSelector.score_model(pd.DataFrame(feat_vec), y, model=model, score_fn=score_fn, cv=cv, find_predicted=False)

            if score_fn == None:
                score = -np.sqrt((scores['test_-RMSE']**2).mean())
            else:
                score = scores['test_user'].mean()
            feat_scores.append(score)

        feat_scores = np.array(feat_scores)

        logger.info('Best score: {0}'.format(feat_scores.max()))
        logger.info('Best feat: {0}'.format(feat_df.columns[np.argmax(feat_scores)]))
        return feat_scores

    @staticmethod
    def score_model(X, y, model=linear_model.LinearRegression(), score_fn=None,
                    cv=None, find_predicted=True, print_scores=False):

        predicted = None
        if cv is None:
            cv = len(y)        # Leave one out

        if score_fn is not None:
            scoring = {#'R2': make_scorer(metrics.r2_score),        # R2 in cross validation is suspect
                       '-RMSE': make_scorer(lambda y, pred: -np.sqrt(metrics.mean_squared_error(y, pred))),
                       'user': make_scorer(score_fn)}
        else:
            scoring = {'-RMSE': make_scorer(lambda y, pred: -np.sqrt(metrics.mean_squared_error(y, pred)))}

        # TO DO: this does k-fold.  We should try stratified k-fold with degradation strata.
        scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)

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




# params: calib_plots from >1 data set
#       model_data_plots from >1 data set
#       (for now the above are the same thing)
#       1 data_set is specified as fitted one, the rest are tests, this can also be done sort of cross-validated
#       a model spec i.e. feature indices and model type
#       num calib plots to use


class ApplyLinearModel(object):
    def __init__(self, in_file_name='', out_file_name='', model=linear_model.LinearRegression, model_keys=[],
                 save_feats=False):
        self.in_file_name = in_file_name
        self.out_file_name = out_file_name
        self.model = model
        self.model_keys = model_keys
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
        pan_bands, band_dict = MsImageFeatureExtractor.get_band_info(num_bands)

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
        if True:
            shape = a.shape[:-1] + (int(1 + (a.shape[-1] - window) / step_size), window)
            strides = a.strides[:-1] + (step_size * a.strides[-1], a.strides[-1])
        else:
            shape = a.shape[:-1] + (window, int(1 + (a.shape[-1] - window) / step_size))
            strides = a.strides[:-1] + (a.strides[-1], step_size * a.strides[-1])
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

    def feat_ex(self, im_buf=[]):
        return

    def create(self, win_size=(1, 1), step_size=(1, 1)):
        with rasterio.Env():
            with rasterio.open(self.in_file_name, 'r') as in_ds:
                # pan_bands, band_dict = MsPatchFeatureExtractor.get_band_info(in_ds.count)
                patch_feature_extractor = MsPatchFeatureExtractor(num_bands=in_ds.count, apply_rolling_window=True,
                                                                  rolling_window_xsize=win_size[0], rolling_window_xstep=step_size[0])
                win_off = np.floor(np.array(win_size) / (2 * step_size[0])).astype('int32')
                prog_update = 10

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
                    for cy in range(0, in_ds.height - win_size[1] + 1, step_size[1]):     #12031): #
                        # read row of windows into mem and slide the win on this rather than the actual file
                        # NB rasterio index is x,y, numpy index is row, col (i.e. y,x)

                        # in_buf = np.zeros((win_size[1], in_ds.width), dtype=in_ds.dtypes[0])
                        in_win = Window(0, cy, out_size[0]*win_size[0], win_size[1])
                        bands = list(range(1, in_ds.count + 1))
                        in_buf = in_ds.read(bands, window=in_win).astype(rasterio.float32)  # NB bands along first dim

                        # TO DO:  deal with -ve vals i.e. make all nans, mask out or something
                        pan = in_buf[patch_feature_extractor._pan_bands, :, :].mean(axis=0)
                        # agc_buf = self.model.intercept_ * np.ones((1, in_ds.width - win_size[0] + 1), dtype=out_ds.dtypes[0])
                        agc_buf = self.model.intercept_ * np.ones((out_size[0]), dtype=out_ds.dtypes[0])
                        out_win = Window(win_off[0], int(cy/step_size[0]) + win_off[1], out_size[0], 1)
                        # for i, (win_fn, band_ratio_fn) in enumerate(zip(self.win_fn_list, self.band_ratio_list)):
                        if False:
                            in_nan_mask = np.any(in_buf <= 0, axis=0) | np.any(pan == 0,
                                                                               axis=0)  # this is overly conservative but neater/faster
                            in_buf[:, in_nan_mask] = np.nan
                            pan[in_nan_mask] = np.nan
                        else:
                            in_nan_mask = np.all(in_buf > 0, axis=0) & np.all(pan != 0,
                                                                               axis=0)  # this is overly conservative but neater/faster
                            feat_dict = patch_feature_extractor.extract_features(in_buf, mask=in_nan_mask, fn_keys=self.model_keys)

                        for i, model_key in enumerate(self.model_keys):
                            if False:
                                feat_buf = patch_feature_extractor.fn_dict[model_key](pan, in_buf) * self.model.coef_[i]
                                agc_buf += feat_buf
                            else:
                                feat_buf = feat_dict[model_key]
                                agc_buf += feat_buf * self.model.coef_[i]
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

                        # TODO: proper progress bar
                        if np.floor(100*cy/ in_ds.height) > prog_update:
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