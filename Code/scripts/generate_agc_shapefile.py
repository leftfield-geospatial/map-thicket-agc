"""

"""
import numpy as np
import geopandas as gpd, pandas as pd
import pathlib, sys, os, glob, warnings

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[2]
else:
    root_path = pathlib.Path(os.getcwd()).parents[0]

sys.path.append(str(root_path.joinpath('Code')))

# Most DGSPS plot locations were corrected / post-processed to ~30cm accuracy = corr_*,
#  some could not be post-processed and are corrected manually here using GCPs = uncorr_*
corr_plot_loc_root_path = root_path.joinpath(r'Data\Sampling Inputs\Plot locations\Corrected')
uncorr_plot_loc_root_path = root_path.joinpath(r'Data\Sampling Inputs\Plot locations\Uncorrected\March 2019')

corr_shapefile_names = [sub_item.joinpath('Point_ge.shp') for sub_item in corr_plot_loc_root_path.iterdir() if sub_item.is_dir()]   # corrected dgps locs
uncorr_shapefile_names = [pathlib.Path(p) for p in glob.glob(str(uncorr_plot_loc_root_path.joinpath('GEF_FIELD*.shp')))]            # uncorrected locs
gcp_shapefile_name = uncorr_plot_loc_root_path.joinpath('GeomaxFieldReferencePts.shp')
plot_agc_allom_filename = root_path.joinpath(r'Data\Outputs\Allometry\Plot AGC.csv')
plot_agc_shapefile_name = root_path.joinpath(r'Data\Outputs\Geospatial\GEF Plot Polygons with AGC v2.shp')

if not plot_agc_allom_filename.exists():
    warnings.warn(f'{plot_agc_allom_filename} does not exist.  You need to run generate_agc_ground_truth')

# read corrected locations
corr_plot_loc_gdf = pd.concat([gpd.read_file(str(corr_shapefile_name)) for corr_shapefile_name in corr_shapefile_names])
corr_plot_loc_gdf['PlotName'] = corr_plot_loc_gdf['Datafile'].str[:-4].str.replace('_0','')
corr_plot_loc_gdf['ID'] = corr_plot_loc_gdf['PlotName'] + '-' + corr_plot_loc_gdf['Comment']
corr_plot_loc_gdf = corr_plot_loc_gdf.set_crs(epsg=4326)        # WGS84
pt_idx = corr_plot_loc_gdf['ID'].str.contains('-H') & ~corr_plot_loc_gdf['ID'].str.contains('FPP')
corr_plot_loc_gdf = corr_plot_loc_gdf[pt_idx][['ID', 'PlotName', 'geometry', 'Comment']]
# read (corrected) gcps for uncorrected locations
gcp_gdf = gpd.read_file(gcp_shapefile_name)

# read and correct uncorrected geomax locations
uncorr_shapefile_name = uncorr_shapefile_names[0]
for uncorr_shapefile_name in uncorr_shapefile_names:
    uncorr_plot_loc_gdf = gpd.read_file(uncorr_shapefile_name)
    pt_idx = uncorr_plot_loc_gdf['NAME'].str.contains('_H') & ~uncorr_plot_loc_gdf['NAME'].str.contains('REF')  # only keep plot corner points
    uncorr_plot_loc_gdf['ID'] = uncorr_plot_loc_gdf['NAME'].str.replace('_H', '-H').str.replace('_','')
    uncorr_plot_loc_gdf['PlotName'] = [id[:id.find('-H')] for id in uncorr_plot_loc_gdf['ID']]
    uncorr_plot_loc_gdf['Comment'] = [id[id.find('-H')+1:] for id in uncorr_plot_loc_gdf['ID']]
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
    # tmp_gdf.to_file(str(uncorr_shapefile_name)[:-4] + '_Corrected.shp')

    corr_plot_loc_gdf = corr_plot_loc_gdf.append(tmp_gdf, ignore_index=True)

# read in AGC ground truth
plot_agc_gdf = gpd.GeoDataFrame(pd.read_csv(plot_agc_allom_filename))
plot_agc_gdf['geometry'] = gpd.GeoSeries()

# merge AGC ground truth with plot geometry
for plot_name, plot_group in corr_plot_loc_gdf.groupby('PlotName'):
    if plot_group.shape[0] != 4:
        warnings.warn(f'{plot_name} should contain 4 corner points, but contains {plot_group.shape[0]}')
    plot_polygon = plot_group['geometry'].unary_union.convex_hull   # create a plot polygon from corner points
    plot_agc_idx = plot_agc_gdf['ID'] == plot_name
    plot_agc_gdf['geometry'][plot_agc_idx] = plot_polygon

# drop null geometry rows
plot_agc_gdf = plot_agc_gdf.drop(np.where(plot_agc_gdf['geometry'].isnull())[0], axis=0)

plot_agc_gdf.to_file(plot_agc_shapefile_name)

