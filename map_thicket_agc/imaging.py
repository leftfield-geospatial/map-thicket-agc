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
import os
import warnings
from collections import OrderedDict
import numpy as np
from sklearn import linear_model
import rasterio
from rasterio.features import sieve
from rasterio.windows import Window
from rasterio.mask import raster_geometry_mask
from rasterio import fill
import geopandas as gpd, pandas as pd

logger = get_logger(__name__)

def nanentropy(x, axis: tuple=None):
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
        for axis_slice in range(x.shape[along_axis]):
            if along_axis == 0:
                xa = x[axis_slice,:,:]
            elif along_axis == 1:
                xa = x[:,axis_slice,:]
            elif along_axis == 2:
                xa = x[:,:,axis_slice]
            else:
                raise Exception('along_axis > 2 or < 0')
            e[axis_slice] = nanentropy(xa, axis=None)
        return e

class PatchFeatureExtractor:
    def __init__(self, num_bands=1, apply_rolling_window=False, rolling_window_xsize=None, rolling_window_xstep=None):
        """
        Virtual class for extracting features from image patches, that hides extraction details from user.
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
        if apply_rolling_window:
            if rolling_window_xsize is None or rolling_window_xstep is None:
                raise Exception("rolling_window_xsize and rolling_window_xstep must be specified")
        self._num_bands = num_bands
        self._apply_rolling_window = apply_rolling_window
        self._rolling_window_xsize = rolling_window_xsize
        self._rolling_window_xstep = rolling_window_xstep
        # self._pan_bands = None
        self._generate_fn_dict()

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

    def _generate_fn_dict(self):
        """
        Generate feature extraction dictionary self.fn_dict with values= pointer to feature extraction functions, and
        keys= descriptive feature strings.  Suitable for use with modelling from patches or mapping a whole image i.e.
        ImageFeatureExtractor and ImageMapper.
        """
        raise NotImplementedError()

    def extract_features(self, im_patch, mask=None, fn_keys=None):
        """
        Virtual method to extract features from image patch with optional mask
        """
        raise NotImplementedError()


class MsPatchFeatureExtractor(PatchFeatureExtractor):
    # TODO: PatchFeatureExtractor and MsPatchFeatureExtractor may benefit from refactoring that gets around the
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
        self._pan_bands, self._band_dict = self.get_band_info(num_bands)
        PatchFeatureExtractor.__init__(self, num_bands=num_bands, apply_rolling_window=apply_rolling_window, rolling_window_xsize=rolling_window_xsize,
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

    def _generate_fn_dict(self):
        """
        Generate feature extraction dictionary with values= pointers to feature extraction functions, and
        keys= descriptive feature strings.
        Suitable for use with modelling from patches or mapping a whole image i.e. ImageFeatureExtractor and ImageMapper.
        Generates band ratios, veg. indices, simple texture measures, and non-linear transformations thereof.
        """

        if len(self.fn_dict) > 0:
            return

        # inner band ratios
        self.inner_dict = OrderedDict()
        self.inner_dict['pan/1'] = lambda pan, bands: pan
        for _num_key in list(self._band_dict.keys()):
            self.inner_dict['{0}/pan'.format(_num_key)] = lambda pan, bands, num_key=_num_key: bands[self._band_dict[num_key], :] / pan
            self.inner_dict['pan/{0}'.format(_num_key)] = lambda pan, bands, num_key=_num_key: pan / bands[self._band_dict[num_key], :]
            self.inner_dict['{0}/1'.format(_num_key)] = lambda pan, bands, num_key=_num_key: bands[self._band_dict[num_key], :]
            for _den_key in list(self._band_dict.keys()):
                if not _num_key == _den_key:
                    self.inner_dict['{0}/{1}'.format(_num_key, _den_key)] = lambda pan, bands, num_key=_num_key, den_key=_den_key:\
                        bands[self._band_dict[num_key], :] / bands[self._band_dict[den_key], :]

        # inner veg indices
        SAVI_L = 0.05   # wikipedia
        nir_keys = [key for key in list(self._band_dict.keys()) if ('NIR' in key) or ('RE' in key)]
        for _nir_key in nir_keys:
            post_fix = '' if _nir_key == 'NIR' else '_{0}'.format(_nir_key)
            self.inner_dict['NDVI' + post_fix] = lambda pan, bands, nir_key=_nir_key: 1 + (bands[self._band_dict[nir_key], :] - bands[self._band_dict['R'], :]) / \
                                                                                     (bands[self._band_dict[nir_key], :] + bands[self._band_dict['R'], :])
            self.inner_dict['SAVI' + post_fix] = lambda pan, bands, nir_key=_nir_key: 1 + (1 + SAVI_L) * (bands[self._band_dict[nir_key], :] - bands[self._band_dict['R'], :]) / \
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
        for inner_key, _inner_fn in self.inner_dict.items():
            for win_key, _win_fn in self.win_dict.items():
                fn_key = '({0}({1}))'.format(win_key, inner_key)
                fn = lambda pan, bands, win_fn=_win_fn, inner_fn=_inner_fn: win_fn(inner_fn(pan, bands))
                self.fn_dict[fn_key] = fn
                if win_key == 'mean':   # for backward compatibility - TODO: try apply scaling to all windows, not only mean
                    for scale_key, _scale_fn in self.scale_dict.items():
                        fn_key = '{0}({1}({2}))'.format(scale_key, win_key, inner_key)
                        fn = lambda pan, bands, scale_fn=_scale_fn, win_fn=_win_fn, inner_fn=_inner_fn: scale_fn(win_fn(inner_fn(pan, bands)))
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
        pan_mask = im_patch_mask[self._pan_bands, :].mean(axis=0)

        feat_dict = OrderedDict()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)    # suppress mean of empty slice warning
            for fn_key in fn_keys:
                feat_dict[fn_key] = self.fn_dict[fn_key](pan_mask, im_patch_mask)

        return feat_dict



class ImageFeatureExtractor(object):
    # Note: this could be a function rather than a class but have made it a class in line with PatchFeatureExtractor etc above
    def __init__(self, image_filename=None, plot_data_gdf=gpd.GeoDataFrame(), store_thumbnail=True):
        """
        Virtual base class to extract features from patches (e.g. ground truth plots) in an image

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
        self._patch_feature_extractor = None     # not implemented, to be specified in derived class constructor
        self._store_thumbnail = store_thumbnail

    def __del__(self):
        self._image_reader.close()

    def extract_image_features(self, feat_keys=None):
        """
        Extract features from plot polygons specified by plot_data_gdf in image

        Returns
        -------
        geopandas.GeoDataFrame of features in subindex 'feats' and data in 'data'
        """
        plot_data_gdf = self._plot_data_gdf
        plot_data_gdf = plot_data_gdf.to_crs(self._image_reader.crs)   # convert plot co-ordinates to image projection

        im_plot_data_dict = dict()      # dict to store plot features etc
        im_plot_count = 0
        max_thumbnail_vals = np.zeros(self._image_reader.count)    # max vals for each image band to scale thumbnails
        im_feat_dict = {}
        im_data_dict = {}

        for plot_id, plot in plot_data_gdf.iterrows():     # loop through plot polygons
            # convert polygon to raster mask
            plot_mask, plot_transform, plot_window = raster_geometry_mask(self._image_reader, [plot['geometry']], crop=True,
                                                                          all_touched=False)
            plot_cnrs_pixel =  np.array(plot_window.toranges())
            plot_mask = ~plot_mask  # TODO: can we lose this?

            # check plot window lies inside image
            if not (np.all(plot_cnrs_pixel[1, :] < self._image_reader.width) and np.all(plot_cnrs_pixel[0, :] < self._image_reader.height)
                    and np.all(plot_cnrs_pixel >= 0)):  # and plot.has_key('Yc') and plot['Yc'] > 0.:
                logger.warning(f'Excluding plot {plot["ID"]} - outside image extent')
                continue

            im_buf = self._image_reader.read(window=plot_window)     # read plot ROI from image

            if np.all(im_buf == 0):
                logger.warning(f'Excluding plot {plot["ID"]} - all pixels are zero')
                continue

            plot_mask = plot_mask & np.all(im_buf > 0, axis=0) & np.all(~np.isnan(im_buf), axis=0)  # exclude any nan or -ve pixels

            im_feat_dict = self._patch_feature_extractor.extract_features(im_buf, mask=plot_mask, fn_keys=feat_keys)  # extract image features for this plot

            # create plot thumbnail for visualisation in numpy format
            if self._store_thumbnail:
                thumbnail = np.float32(np.moveaxis(im_buf, 0, 2))
                thumbnail[~plot_mask] = 0.
                im_data_dict = {**im_feat_dict, **plot, 'thumbnail': thumbnail}  # combine features and other plot data

                # calc max thumbnail vals for scaling later
                max_val = np.percentile(thumbnail, 98., axis=(0, 1))
                max_thumbnail_vals[max_val > max_thumbnail_vals] = max_val[max_val > max_thumbnail_vals]
            else:
                im_data_dict = {**im_feat_dict, **plot}                             # combine features and other plot data

            im_plot_data_dict[plot_id] = im_data_dict       # add to dict of all plots
            im_plot_count += 1

            log_dict = {'ABC': 'Abc' in plot, 'Num zero pixels': (im_buf == 0).sum(), 'Num -ve pixels': (im_buf < 0).sum(),
                'Num nan pixels': np.isnan(im_buf).sum()}
            logger.info(', '.join([f'Plot {plot_id}'] + ['{}: {}'.format(k, v) for k, v in log_dict.items()]))

        logger.info('Processed {0} plots'.format(im_plot_count))

        # scale thumbnails for display
        if self._store_thumbnail:
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
        self.im_plot_data_gdf = self.im_plot_data_gdf.set_crs(self._image_reader.crs)
        self.im_plot_data_gdf.columns = columns
        self.im_plot_data_gdf[('data','ID')] = self.im_plot_data_gdf.index

        return self.im_plot_data_gdf


class MsImageFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, image_filename=None, plot_data_gdf=gpd.GeoDataFrame(), store_thumbnail=True):
        """
        Class to extract multi-spectral features from patches (e.g. ground truth plots) in an image

        Parameters
        ----------
        image_filename : str
            path to image file from which to extract features
        plot_data_gdf : geopandas.GeoDataFrame
            plot polygons with optional ground truth and index of plot ID strings
        """
        ImageFeatureExtractor.__init__(self, image_filename=image_filename, plot_data_gdf=plot_data_gdf, store_thumbnail=store_thumbnail)
        self._patch_feature_extractor = MsPatchFeatureExtractor(num_bands=self._image_reader.count)


class ImageMapper(object):
    def __init__(self, image_file_name='', map_file_name='', model=linear_model.LinearRegression(), model_feat_keys: list=None,
                 save_feats=False):
        """
        Virtual base class to generate raster map by applying a fitted model to an image

        Parameters
        ----------
        image_file_name : str
            path to input image file
        map_file_name: str
            output map file path to create
        model : sklearn.BaseEstimator
            trained/ fitted model to apply to image_file_name
        model_feat_keys : list
            list of descriptive strings specifying features to extract from image_file_name and to feed to model,
            (as understood by feature_extractor_type)
        save_feats : bool
            include extracted features as bands in map_file_name (default = False)
        """
        self.image_file_name = image_file_name
        self.map_file_name = map_file_name
        self.model = model
        self.model_keys = model_feat_keys
        self.save_feats = save_feats
        self.nodata = np.nan        # nodata value to use for map raster
        self._feature_extractor_type = PatchFeatureExtractor    # not implemented, to be specified in derived class

    def map(self, win_size=(1, 1), step_size=(1, 1)):
        """
        Apply model to image features and write to map raster file

        Parameters
        ----------
        win_size : numpy.array_like
            (x, y) rolling window size in pixels (default = (1,1))
        step_size : numpy.array_like
            (x, y) rolling window steps in pixels (default = (1,1))
        """
        if np.any(np.array(step_size) > np.array(win_size)):
            logger.warning('step_size <= win_size')

        with rasterio.Env():
            with rasterio.open(self.image_file_name, 'r') as image:
                # create the feature extraction class
                feature_extractor = self._feature_extractor_type(num_bands=image.count, apply_rolling_window=True,
                                                                          rolling_window_xsize=win_size[0], rolling_window_xstep=step_size[0])
                map_win_off = np.array([0, 0])  # test
                prog_update = 10

                if self.save_feats:
                    out_bands = len(self.model_keys) + 1
                else:
                    out_bands = 1

                # setup the output raster metadata based on image metadata and specified win_size and step_size
                map_profile = image.profile
                map_size = np.floor([1 + (image.width - win_size[0])/step_size[0], 1 + (image.height - win_size[1])/step_size[1]]).astype('int32')
                # map_size = np.floor([image.width / step_size[0], image.height / step_size[1]]).astype('int32')
                map_profile.update(dtype=rasterio.float32, count=out_bands, compress='deflate', driver='GTiff', width=map_size[0],
                                   height=map_size[1], nodata=self.nodata)

                map_profile['transform'] = map_profile['transform'] * rasterio.Affine.scale(*step_size)
                map_profile['transform'] = map_profile['transform'] * rasterio.Affine.translation(*np.floor(np.array(win_size)/(2 * np.array(step_size))))
                if (map_size[0] / map_profile['blockxsize'] < 10) | (map_size[1] / map_profile['blockysize'] < 10):
                    map_profile['tiled'] = False
                    map_profile['blockxsize'] = map_size[0]
                    map_profile['blockysize'] = 1

                with rasterio.open(self.map_file_name, 'w', **map_profile) as map_im:
                    # read image by groups of N rows where N = win_size[1], then slide window in x direction
                    # Note: rasterio index is x,y, numpy index is row, col (i.e. y,x)
                    for cy in range(0, image.height - win_size[1] + 1, step_size[1]):
                        image_win = Window(0, cy, (map_size[0] - 1)*step_size[0] + win_size[0], win_size[1])
                        bands = list(range(1, image.count + 1))
                        image_buf = image.read(bands, window=image_win).astype(rasterio.float32)  # NB bands along first dim

                        # pan = image_buf[feature_extractor._pan_bands, :, :].mean(axis=0)
                        map_win = Window(map_win_off[0], int(cy/step_size[0]) + map_win_off[1], map_size[0], 1)

                        # mask out negative and zero values to prevent NaN outputs - overly conservative but neater & faster than other options
                        image_nan_mask = np.all(image_buf > 0, axis=0)  # & np.all(pan != 0, axis=0)
                        # extract features (model features only - not entire library) from image_buf using rolling window
                        feat_dict = feature_extractor.extract_features(image_buf, mask=image_nan_mask, fn_keys=self.model_keys)

                        feat_buf = np.array([fv for fv in feat_dict.values()]).transpose()
                        feat_buf_nan_mask = np.any(np.isnan(feat_buf) | np.isinf(feat_buf), axis=1)    # work around for nan's which raise an error in sklearn
                        feat_buf[feat_buf_nan_mask, :] = 0
                        map_buf = self.model.predict(feat_buf).astype(map_im.dtypes[0])
                        map_buf[feat_buf_nan_mask] = self.nodata

                        map_im.write(map_buf.reshape(1, -1), indexes=1, window=map_win)
                        if self.save_feats:
                            feat_buf[feat_buf_nan_mask, :] = self.nodata
                            feat_buf = feat_buf.astype(map_im.dtypes[0]).transpose()[:, np.newaxis, :]   # order axes correctly for rasterio write
                            map_im.write(feat_buf, indexes=np.arange(2, 2 + len(self.model_keys)), window=map_win)

                        # TODO: proper progress bar
                        if np.ceil(100 * cy / (image.height - win_size[1] + 1)) >= prog_update:
                            logger.info(f'Progress {prog_update}%')
                            prog_update += 10

class MsImageMapper(ImageMapper):
    def __init__(self, image_file_name='', map_file_name='', model=linear_model.LinearRegression, model_feat_keys: list=None,
                 save_feats=False):
        """
        Class to generate raster map by applying fitted model to a multispectral image

        Parameters
        ----------
        image_file_name : str
            path to input image file
        map_file_name: str
            output map file path to create
        model : sklearn.BaseEstimator
            trained/ fitted model to apply to image_file_name
        model_feat_keys : list
            list of descriptive strings specifying features to extract from image_file_name and to feed to model,
            (as understood by feature_extractor_type)
        save_feats : bool
            include extracted features as bands in map_file_name (default = False)
        """
        ImageMapper.__init__(self, image_file_name=image_file_name, map_file_name=map_file_name, model=model, model_feat_keys=model_feat_keys,
                 save_feats=save_feats)
        self._feature_extractor_type = MsPatchFeatureExtractor

def thicket_agc_post_proc(image_mapper = ImageMapper()):
    """
    Post process thicket AGC map - helper function for MsImageMapper
    Writes out a cleaned version of map generated by ImageMapper.map(...), ills nodata and places sensible limits on
    AGC values.

    Parameters
    ----------
    image_mapper : ImageMapper
        instance of ImageMapper that has generated map

    Returns
    -------
    post processed raster file name
    """
    with rasterio.Env():
        with rasterio.open(image_mapper.map_file_name, 'r') as in_ds:
            out_profile = in_ds.profile
            out_profile.update(count=1)
            split_ext = os.path.splitext(image_mapper.map_file_name)
            out_file_name = '{0}_postproc{1}'.format(split_ext[0], split_ext[1])
            with rasterio.open(out_file_name, 'w', **out_profile) as out_ds:
                if (not out_profile['tiled']) or (np.prod(in_ds.shape) < 10e6):
                    in_windows = enumerate([Window(0,0,in_ds.width, in_ds.height)])    # read whole raster at once
                else:
                    in_windows = in_ds.block_windows(1)                                # read in blocks

                for ji, block_win in in_windows:
                    in_block = in_ds.read(1, window=block_win, masked=True)

                    in_block[in_block < 0] = 0
                    in_block.mask = (in_block.mask.astype(np.bool) | (in_block > 95) | (in_block < 0)).astype(rasterio.uint8)

                    in_mask = in_block.mask.copy()
                    sieved_msk = sieve(in_mask.astype(rasterio.uint8), size=2000)

                    out_block = fill.fillnodata(in_block, mask=None, max_search_distance=20, smoothing_iterations=1)
                    out_block[sieved_msk.astype(np.bool)] = image_mapper.nodata
                    out_ds.write(out_block, indexes=1, window=block_win)
        return out_file_name
