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
from map_thicket_agc import root_path
from glob import glob
from matplotlib import pyplot
from pathlib import Path
import numpy as np
import geopandas as gpd
from map_thicket_agc.imaging import MsImageFeatureExtractor
from map_thicket_agc.visualisation import scatter_ds
from pprint import pprint
from scipy import stats

# -----------------------------------------------------------------------------
# compare performance of different references and different homonim fuse params
image_root_path = root_path.joinpath(r'data/inputs/imagery')
plot_agc_shapefile_name = root_path.joinpath(r'data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp')
src_file = image_root_path.joinpath(r'NGI/Ngi_May2015_OrthoNgiDem_Source_Mosaic.vrt')

# create a list of all the available corrected VRT mosaic files
corrected_wildcards = [
    r'V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Corrected\Landsat-8\*.vrt',
    r'V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Corrected\Sentinel-2\*.vrt',
    r'V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Corrected\Sentinel-2-Harm\*.vrt',
    r'V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Corrected\Modis-nbar\*.vrt',
]

corrected_files = []
for corrected_wildcard in corrected_wildcards:
    corrected_files += [*glob(corrected_wildcard)]
pprint(corrected_files)

def correct_stratum(gdf):
    # fix stratum labels
    gdf.loc[gdf['data']['Stratum'] == 'Degraded', ('data', 'Stratum')] = 'Severe'
    return gdf

# load AGC ground truth shapefile
plot_agc_gdf = gpd.GeoDataFrame.from_file(plot_agc_shapefile_name)
plot_agc_gdf = plot_agc_gdf.set_index('ID').sort_index()

# Extract features from the source (uncorrected) images
source_fex = MsImageFeatureExtractor(image_filename=src_file, plot_data_gdf=plot_agc_gdf)
source_gdf = source_fex.extract_image_features()
source_gdf = correct_stratum(source_gdf)

# Extract features from the corrected VRT mosaics
corrected_dict = {}
for corrected_file in corrected_files:
    print(f'Extracting features for {corrected_file}')
    corrected_fex = MsImageFeatureExtractor(image_filename=corrected_file, plot_data_gdf=plot_agc_gdf)
    corrected_gdf = corrected_fex.extract_image_features()
    corrected_gdf = correct_stratum(corrected_gdf)
    corrected_dict[str(corrected_file)] = corrected_gdf

# Find R2 correlation coefficient for each of the corrected VRT mosaics
feat = '(mean(NDVI))'  # this is one of the better performing features and commonly understood
corrected_r2 = {}
for corrected_gdf, corrected_file in zip(corrected_dict.values(), corrected_files):
    cc = np.corrcoef(corrected_gdf[('feats', feat)], corrected_gdf[('data', 'AgcHa')]/1000)
    corrected_r2[str(corrected_file)] = cc[0, 1] ** 2

pprint(corrected_r2)
# features corresponding to the best R2
corrected_gdf = corrected_dict[max(corrected_r2, key=corrected_r2.get)]

def plot_agc_corr(x, y, x_label='NDVI', y_label='AGC (t C ha$^{-1}$)'):
    """ Plot ground truth vs feature vals with R2 text. """
    xlim = [np.nanmin(x), np.nanmax(x)]
    ylim = [np.nanmin(y), np.nanmax(y)]
    xd = np.diff(xlim)[0]
    yd = np.diff(ylim)[0]

    pyplot.axis('tight')
    pyplot.axis(xlim + ylim)
    pyplot.plot(x, y, marker='.', linestyle='None', markersize=7)

    if True:
        (slope, intercept, r, p, stde) = stats.linregress(x, y)
        pyplot.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(r ** 2),
                    fontdict={'size': 12})
        yr = np.array(xlim) * slope + intercept
        pyplot.plot(xlim, yr, 'k--', lw=2, zorder=-1)

    pyplot.xlabel(x_label, fontdict={'size': 12})
    pyplot.ylabel(y_label, fontdict={'size': 12})
    # pyplot.axis('tight')

# create before and after correction scatter plots
labels = ['Source', 'Corrected']
fig = pyplot.figure()
fig.set_size_inches(10, 4.5, forward=True)

for i, gdf in enumerate([source_gdf, corrected_gdf]):
    pyplot.subplot(1, 2, i+1)
    plot_agc_corr(gdf[('feats', feat)], gdf[('data', 'AgcHa')]/1000)
    pyplot.title(labels[i])

# pyplot.savefig(root_path.joinpath(f'data/outputs/plots/homonim_ngi_case_study.png'), dpi=300)

#---------------------------------------------------------------------------------------------------------------
# show image mosaics before and after correction
%matplotlib
import rasterio as rio
from rasterio.plot import show
from rasterio.enums import Resampling
from matplotlib import pyplot
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from map_thicket_agc import root_path

image_root_path = root_path.joinpath(r'data/inputs/imagery')
src_file = image_root_path.joinpath(r'NGI/Ngi_May2015_OrthoNgiDem_Source_Mosaic.vrt')
corr_file = image_root_path.joinpath(r'NGI/Ngi_May2015_OrthoNgiDem_Corrected_Mosaic.vrt')
plot_agc_shapefile_name = root_path.joinpath(r'data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp')

plot_agc_gdf = gpd.GeoDataFrame.from_file(plot_agc_shapefile_name)

indexes = [1, 2, 3]
ds_fact = 4  # downsample factor


fig, ax = pyplot.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.1)

with rio.Env(GDAL_NUM_THREADS='ALL_CPUs', GTIFF_FORCE_RGBA=False), rio.open(corr_file, 'r') as ds:
    ds_shape = tuple(np.round(np.array(ds.shape) / ds_fact).astype(int).tolist())
    array = ds.read(indexes=indexes, out_dtype='float32', out_shape=ds_shape)  # , resampling=Resampling.average)
    mask = np.any(array == ds.nodata, axis=(0)) | np.any(np.isnan(array), axis=(0))
    array[:, mask] = np.nan
    for bi in range(len(indexes)):
        array[bi] -= np.nanpercentile(array[bi], 2)
        array[bi] /= np.nanpercentile(array[bi], 98)
        array[bi] = np.clip(array[bi], 0, 1)

    transform = ds.transform * rio.Affine.scale(ds_fact)
    ax = show(array, transform=transform, interpolation='bilinear', ax=ax)
    _plot_agc_gdf = plot_agc_gdf.to_crs(ds.crs)
    _plot_agc_gdf.geometry = _plot_agc_gdf.geometry.centroid
    _plot_agc_gdf.AgcHa /= 1000
    ax = _plot_agc_gdf.plot(
        'AgcHa', kind='geo', legend=True, ax=ax, cmap='RdYlGn', cax=cax, edgecolor='white', linewidth=0.5,
        legend_kwds=dict(label='Aboveground Carbon (t C ha$^{-1}$)', orientation='vertical')
    )
    ax.axis((86494.06047619047, 94313.07562770562, -3717680.7510822513, -3711286.8766233767))
    ax.axis('off')

