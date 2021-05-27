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

##
from map_thicket_agc import root_path
import numpy as np
import geopandas as gpd, pandas as pd
import pathlib, glob
from map_thicket_agc import get_logger

logger = get_logger(__name__)

corr_plot_loc_root_path = root_path.joinpath(r'data/inputs/geospatial/sampling_plot_locations/corrected')
uncorr_plot_loc_root_path = root_path.joinpath(r'data/inputs/geospatial/sampling_plot_locations/uncorrected/march_2019')

corr_shapefile_names = [sub_item.joinpath('Point_ge.shp') for sub_item in corr_plot_loc_root_path.iterdir() if sub_item.is_dir()]   # corrected dgps locs
uncorr_shapefile_names = [pathlib.Path(p) for p in glob.glob(str(uncorr_plot_loc_root_path.joinpath('GEF_FIELD*.shp')))]            # uncorrected locs
gcp_shapefile_name = uncorr_plot_loc_root_path.joinpath('geomax_field_reference_pts.shp')
plot_agc_allom_filename = root_path.joinpath(r'data/outputs/allometry/plot_agc_v3.csv')
plot_agc_shapefile_name = root_path.joinpath(r'data/outputs/geospatial/gef_plot_polygons_with_agc_v2.shp')

## manually correct DGPS plot locations that could not be post-processed
logger.info('Starting...')
if not plot_agc_allom_filename.exists():
    logger.warning(f'{plot_agc_allom_filename} does not exist.  You need to run generate_agc_ground_truth')

# read corrected (trimble) locations
corr_plot_loc_gdf = pd.concat([gpd.read_file(str(corr_shapefile_name)) for corr_shapefile_name in corr_shapefile_names])
corr_plot_loc_gdf['PlotName'] = corr_plot_loc_gdf['Datafile'].str[:-4].str.replace('_0','')
corr_plot_loc_gdf['ID'] = corr_plot_loc_gdf['PlotName'] + '-' + corr_plot_loc_gdf['Comment']
corr_plot_loc_gdf = corr_plot_loc_gdf.set_crs(epsg=4326)        # WGS84
pt_idx = corr_plot_loc_gdf['ID'].str.contains('-H') & ~corr_plot_loc_gdf['ID'].str.contains('FPP')
corr_plot_loc_gdf = corr_plot_loc_gdf[pt_idx][['ID', 'PlotName', 'geometry', 'Comment']]

gcp_gdf = gpd.read_file(gcp_shapefile_name) # read (corrected) gcps for uncorrected locations

# read uncorrected (geomax) locations, and correct
uncorr_shapefile_name = uncorr_shapefile_names[0]
for uncorr_shapefile_name in uncorr_shapefile_names:
    uncorr_plot_loc_gdf = gpd.read_file(uncorr_shapefile_name)
    pt_idx = uncorr_plot_loc_gdf['NAME'].str.contains('_H') & ~uncorr_plot_loc_gdf['NAME'].str.contains('REF')  # only keep plot corner points
    uncorr_plot_loc_gdf['ID'] = uncorr_plot_loc_gdf['NAME'].str.replace('_H', '-H').str.replace('_','')
    uncorr_plot_loc_gdf['PlotName'] = [plot_id[:plot_id.find('-H')] for plot_id in uncorr_plot_loc_gdf['ID']]
    uncorr_plot_loc_gdf['Comment'] = [plot_id[plot_id.find('-H') + 1:] for plot_id in uncorr_plot_loc_gdf['ID']]
    if uncorr_plot_loc_gdf.crs is None:
        uncorr_plot_loc_gdf = uncorr_plot_loc_gdf.set_crs(gcp_gdf.crs)

    # match GCPs in uncorr_plot_gdf to gcp_gdf
    gcp_ids, uncorr_idx, gcp_idx = np.intersect1d(uncorr_plot_loc_gdf['NAME'], gcp_gdf['ID'], return_indices=True)

    # Calculate correction offset - No high level way of converting GeoSeries to numpy arrays, and or doing array type arithmetic on GeoSeries, so ...
    offsets = np.array([(np.array(uncorr_gcp_pt)[:2] - np.array(gcp_pt)[:2]).tolist()
                        for gcp_pt, uncorr_gcp_pt in zip(uncorr_plot_loc_gdf.iloc[uncorr_idx]['geometry'], gcp_gdf.iloc[gcp_idx]['geometry'])])
    offset = np.mean(offsets, axis=0)

    # apply correction to new dataframe
    tmp_gdf = uncorr_plot_loc_gdf[pt_idx][['ID', 'PlotName', 'geometry', 'Comment']]
    tmp_gdf['geometry'] = uncorr_plot_loc_gdf[pt_idx]['geometry'].translate(xoff=offset[0], yoff=offset[1])

    tmp_gdf = tmp_gdf.to_crs(corr_plot_loc_gdf.crs) # convert to corr_plot_loc_gdf CRS (does not work with geopandas from https://www.lfd.uci.edu/~gohlke/pythonlibs/ - use conda)
    corr_plot_loc_gdf = corr_plot_loc_gdf.append(tmp_gdf, ignore_index=True)

# read in AGC ground truth
plot_agc_gdf = gpd.GeoDataFrame(pd.read_csv(plot_agc_allom_filename))
plot_agc_gdf['geometry'] = gpd.GeoSeries()
plot_agc_gdf = plot_agc_gdf.set_crs(corr_plot_loc_gdf.crs)

# merge AGC ground truth with plot geometry
for plot_name, plot_group in corr_plot_loc_gdf.groupby('PlotName'):
    if plot_group.shape[0] != 4:
        logger.warning(f'{plot_name} should contain 4 corner points, but contains {plot_group.shape[0]}')
    plot_polygon = plot_group['geometry'].unary_union.convex_hull   # create a plot polygon from corner points
    plot_agc_idx = plot_agc_gdf['ID'] == plot_name
    plot_agc_gdf['geometry'][plot_agc_idx] = plot_polygon

# drop null geometry rows
plot_agc_gdf = plot_agc_gdf.drop(np.where(plot_agc_gdf['geometry'].isnull())[0], axis=0)
plot_agc_gdf.to_file(plot_agc_shapefile_name)   # write shapefile

logger.info('Done\n')
if __name__ =='__main__':
    input('Press ENTER to continue...')
