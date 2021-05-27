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
## Generate model calibration and AGC data for https://github.com/dugalh/extend_thicket_agc

import geopandas as gpd, pandas as pd
import numpy as np
from map_thicket_agc import imaging as img
from map_thicket_agc import get_logger
from map_thicket_agc import root_path
import joblib

image_root_path = root_path.joinpath(r'data/inputs/imagery')
calib_plot_in_file = root_path.joinpath(r'data/inputs/geospatial/gef_calib_plots.shp')
calib_plot_out_file = root_path.joinpath(r'data/outputs/geospatial/gef_calib_plots_with_agc.geojson')
calib_plot_out_file_translated = root_path.joinpath(r'data/outputs/geospatial/gef_calib_plots_with_agc_translated.geojson')

image_file = image_root_path.joinpath(r'WorldView3_Oct2017_OrthoNgiDem_AtcorSrtmAdjCorr_PanAndPandSharpMs.tif')
model_file = root_path.joinpath(r'data/outputs/Models/best_univariate_model_py38_cv5v2.joblib')

feats_of_interest = ['log(mean(R/pan))', 'log(mean(G/R))', 'log(mean(R/NIR))', '(mean(NDVI))', '(mean(SAVI))', 'log(mean(B/R))']

logger = get_logger(__name__)
logger.info('Starting...')

# extract feats_of_interest for plots in calib_plot_in_file
calib_plot_gdf = gpd.GeoDataFrame.from_file(calib_plot_in_file)
fex = img.MsImageFeatureExtractor(image_file, plot_data_gdf=calib_plot_gdf, store_thumbnail=False)
im_calib_plot_gdf = fex.extract_image_features(feat_keys=feats_of_interest)

gdf = pd.concat([im_calib_plot_gdf['data'], im_calib_plot_gdf['feats']], axis=1) # flatten for export
model, model_feat_keys, model_scores = joblib.load(model_file)

if not set(model_feat_keys).issubset(feats_of_interest):    # make sure we have found the model features
    raise Exception('feats_of_interest does not contain model_feat_keys')

gdf['AgcHa'] = model.predict(gdf[model_feat_keys])      # model est AGC val, not the allometric val
gdf_translate = gdf.copy(True)
gdf = gdf.to_crs(epsg=4326)
gdf.to_file(calib_plot_out_file, driver='GeoJSON')

# debug code to test correction of landsat spatial offset
gdf_translate['geometry'] = gdf_translate['geometry'].translate(xoff=10, yoff=40)
gdf_translate = gdf_translate.to_crs(epsg=4326)
gdf_translate.to_file(calib_plot_out_file_translated, driver='GeoJSON')

logger.info('Done\n')
if __name__ =='__main__':
    input('Press ENTER to continue...')
