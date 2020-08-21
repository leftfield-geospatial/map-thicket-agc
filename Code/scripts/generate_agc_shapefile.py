"""

"""
import gdal, ogr, osr
import numpy as np
from collections import OrderedDict
import fiona, pyproj
import geopandas as gpd, pandas as pd
from modules import SpatialUtils as su
import pathlib, sys, os, glob
import shapely, fiona

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[2]
else:
    root_path = pathlib.Path(os.getcwd()).parents[0]

sys.path.append(str(root_path.joinpath('Code')))
from modules import SpatialUtils as su

# most DGSPS plot locations were corrected / post-processed to ~30cm accuracy = corr_*
# some could not be post-processed and are corrected manually here using GCPs = uncorr_*
corr_plot_loc_root_path = root_path.joinpath(r'Data\Sampling Inputs\Plot locations\Corrected')
uncorr_plot_loc_root_path = root_path.joinpath(r'Data\Sampling Inputs\Plot locations\Uncorrected\March 2019')

corr_shapefile_names = [sub_item.joinpath('Point_ge.shp') for sub_item in corr_plot_loc_root_path.iterdir() if sub_item.is_dir()]   # corrected dgps locs
uncorr_shapefile_names = [pathlib.Path(p) for p in glob.glob(str(uncorr_plot_loc_root_path.joinpath('GEF_FIELD*.shp')))]            # uncorrected locs
gcp_shapefile_name = uncorr_plot_loc_root_path.joinpath('GeomaxFieldReferencePts.shp')

# read corrected locations
corr_plot_loc_gdf = pd.concat([gpd.read_file(str(corr_shapefile_name)) for corr_shapefile_name in corr_shapefile_names])
corr_plot_loc_gdf['PlotName'] = corr_plot_loc_gdf['Datafile'].str[:-4].str.replace('_0','')
corr_plot_loc_gdf['ID'] = corr_plot_loc_gdf['PlotName'] + '-' + corr_plot_loc_gdf['Comment']
corr_plot_loc_gdf = corr_plot_loc_gdf.set_crs(epsg=4326)        # WGS84
corr_plot_loc_gdf = corr_plot_loc_gdf[['ID', 'PlotName', 'geometry', 'Comment']]
# read (corrected) gcps for uncorrected locations
gcp_gdf = gpd.read_file(gcp_shapefile_name)
# gcp_gdf.to_crs(corr_plot_loc_gdf.crs.to_wkt())  # convert to corr_plot_loc_gdf CRS (does not work with geopandas from https://www.lfd.uci.edu/~gohlke/pythonlibs/ - use conda)

uncorr_shapefile_name = uncorr_shapefile_names[0]
for uncorr_shapefile_name in uncorr_shapefile_names:
    uncorr_plot_loc_gdf = gpd.read_file(uncorr_shapefile_name)
    pt_idx = uncorr_plot_loc_gdf['NAME'].str.contains('_H')
    uncorr_plot_loc_gdf['ID'] = uncorr_plot_loc_gdf['NAME'].str.replace('_H', '-H').str.replace('_','')
    uncorr_plot_loc_gdf['PlotName'] = [id[:id.find('-H')] for id in uncorr_plot_loc_gdf['ID']]
    uncorr_plot_loc_gdf['Comment'] = [id[id.find('-H')+1:] for id in uncorr_plot_loc_gdf['ID']]
    if uncorr_plot_loc_gdf.crs is None:
        uncorr_plot_loc_gdf = uncorr_plot_loc_gdf.set_crs(gcp_gdf.crs)
    # tmp_corr_gdf = uncorr_plot_loc_gdf.copy()

    # match GCPs in uncorr_plot_gdf to gcp_gdf
    gcp_ids, uncorr_idx, gcp_idx = np.intersect1d(uncorr_plot_loc_gdf['NAME'], gcp_gdf['ID'], return_indices=True)

    # No high level way of converting GeoSeries to numpy arrays, and or doing array type arithmetic on GeoSeries,
    #   so we do the below
    offsets = np.array([(np.array(gcp_pt)[:2] - np.array(uncorr_gcp_pt)[:2]).tolist()
                        for gcp_pt, uncorr_gcp_pt in zip(uncorr_plot_loc_gdf.iloc[uncorr_idx]['geometry'], gcp_gdf.iloc[gcp_idx]['geometry'])])
    offset = np.mean(offsets, axis=0)
    tmp_plot_loc_gdf = uncorr_plot_loc_gdf['geometry'].translate(xoff=offset[0], yoff=offset[1])

    # apply GCP offset correction
    tmp_gdf = uncorr_plot_loc_gdf[pt_idx][['ID', 'PlotName', 'geometry', 'Comment']]
    tmp_gdf['geometry'] = uncorr_plot_loc_gdf[pt_idx]['geometry'].translate(xoff=offset[0], yoff=offset[1])
    # corr_geom_gdf.to_file('{0}_Corrected.shp'.format(str(uncorr_shapefile_name)[:-4]))

    corr_plot_loc_gdf = corr_plot_loc_gdf.append(tmp_gdf, ignore_index=True)

# corr_plot_loc_gdf.to_file('C:/Temp/temp.csv', driver='CSV')



    if '_H' in key and not 'REF' in key:
        plot_name = key[:key.find('_H')]
        comment = key[key.find('_H') + 1:]
        point = {}
        point['PlotName'] = str(plot_name).replace('_', '')
        point['Comment'] = comment
        point['geom'] = field_point['geom_corr'].Clone()
        # point['geom'] = field_point['geom'].Clone()
        point['geom'].Transform(geomax_to_dgps_transform)
        point['X'] = point['geom'].GetX()
        point['Y'] = point['geom'].GetX()
        dgpsDict[key] = point

field_readers = []
make_corrected_files = False
for geomax_shapefile_name in geomax_shapefile_names:
    field_reader = su.GdalVectorReader(geomax_shapefile_name)
    field_reader.read(id_field='NAME')
    # hack: srs not set, so set it here
    list(field_reader.layer_dict.values())[0]['spatial_ref'] = gcp_reader.layer_dict['GeomaxFieldReferencePts']['spatial_ref'].Clone()
    field_point_dict = OrderedDict(list(field_reader.layer_dict.values())[0]['feat_dict'])

    # find correction offset for this file (one offset that is the mean of all gcp differences)
    field_point_names = np.array(list(field_point_dict.keys()))
    gcp_idx = np.array([field_point_name.startswith('REF') for field_point_name in field_point_names])
    offsets = []
    for gcp_key in field_point_names[gcp_idx]:
        field_gcp = field_point_dict[gcp_key]
        ref_gcp = ref_gcp_point_dict[gcp_key]
        offset = [ref_gcp['X'] - field_gcp['X'], ref_gcp['Y'] - field_gcp['Y']]
        offsets.append(offset)
    # find one offset that is the mean of all gcp differences
    offset = np.array(offsets).mean(axis=0)
    print(field_point_names[gcp_idx])

    # create output corrected shapefile
    if make_corrected_files:
        corrected_shapefile_name = '{0}_Corrected.shp'.format(geomax_shapefile_name[:-4])
        ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(corrected_shapefile_name)
        # layer = ds.CreateLayer(corrected_shapefile_name[:-4], gcp_reader.layer_dict['GeomaxFieldReferencePts']['spatial_ref'].Clone(), ogr.wkbPoint)
        layer = ds.CreateLayer(corrected_shapefile_name[:-4], dgpsSrs.Clone(), ogr.wkbPoint)
        layer.CreateField(ogr.FieldDefn("ID", ogr.OFTString))

    # Add the fields we're interested in
    # apply the offset to all the points
    for key, point in field_point_dict.items():
        point['X_corr']  = point['X'] + offset[0]
        point['Y_corr'] = point['Y'] + offset[1]
        point_corr = ogr.Geometry(ogr.wkbPoint)
        point_corr.AddPoint(point['X_corr'], point['Y_corr'])
        point['geom_corr'] = point_corr
        # point_corr.Transform(geomax_to_dgps_transform)

        if make_corrected_files:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("ID", key)
            feature.SetGeometry(point_corr)
            layer.CreateFeature(feature)


    # ds = None
    field_readers.append(field_reader)

    # reformat, transform srs and add to dgpsDict
    # 'PlotName': 'TCH_03',
    # 'Comment': 'H1',
    for key, field_point in field_point_dict.items():
        if '_H' in key and not 'REF' in key :
            plot_name = key[:key.find('_H')]
            comment = key[key.find('_H')+1:]
            point = {}
            point['PlotName'] = str(plot_name).replace('_','')
            point['Comment'] = comment
            point['geom'] = field_point['geom_corr'].Clone()
            # point['geom'] = field_point['geom'].Clone()
            point['geom'].Transform(geomax_to_dgps_transform)
            point['X']= point['geom'].GetX()
            point['Y']= point['geom'].GetX()
            dgpsDict[key] = point


# field_reader.layer_dict['GEF_FIELD4']['feat_dict']['REF_Toilet_NW']['geom']

print(list(field_reader.layer_dict['GEF_FIELD4']['feat_dict'].keys()))
print(field_reader.layer_dict['GEF_FIELD4']['spatial_ref'])




## read dgps points into dictionary
dgpsSrs = osr.SpatialReference()
dgpsDict = OrderedDict()
for corr_shapefile_name in corr_shapefile_names:
    # ds = gdal.OpenEx(correctedShapeFileName, gdal.OF_VECTOR)
    # with fiona.open(corr_shapefile_name) as lyr:

    ds = gdal.OpenEx(str(corr_shapefile_name), gdal.OF_VECTOR)
    if ds is None:
        print('Failed to open {0}'.format(corr_shapefile_name))
        break
    ds.ResetReading()

    lyr = ds.GetLayerByIndex(0)
    lyr.ResetReading()
    if lyr.GetSpatialRef() is not None:
        dgpsSrs = lyr.GetSpatialRef()
    else:
        dgpsSrs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    feat_defn = lyr.GetLayerDefn()
    for (i, feat) in enumerate(lyr):
        print('.', end=' ')
        f = {}
        for i in range(feat_defn.GetFieldCount()):
            field_defn = feat_defn.GetFieldDefn(i)
            f[field_defn.GetName()] = feat.GetField(i)

        key = "%s-%s" % (f['Datafile'][:-4], f['Comment'])

        geom = feat.GetGeometryRef()
        if geom is not None and (geom.GetGeometryType() == ogr.wkbPoint or geom.GetGeometryType() == ogr.wkbPointZM):
            # concat(left(Datafile,strpos(Datafile,'.cor')-1), '-',  Comment)
            print("%s: %.6f, %.6f" % (key, geom.GetX(), geom.GetY()))
            f['geom'] = geom.Clone()
            f['X'] = geom.GetX()
            f['Y'] = geom.GetY()
        else:
            print("%s has no point geometry" % (key))
            break
        # gcpList.append(f)
        f['PlotName'] = f['Datafile'][:-4]
        dgpsDict[key] = f
        # feat = None
    print(' ')
    # lyr = None
    # ds = None

## Read the Geomax March 2019 files, correct them (and write out).  Add corrected info to the dgpsDict

gcp_reader = su.GdalVectorReader(gcp_shapefile_name)
gcp_reader.read()

print(list(gcp_reader.layer_dict['GeomaxFieldReferencePts']['feat_dict'].keys()))
print(gcp_reader.layer_dict['GeomaxFieldReferencePts']['spatial_ref'])
ref_gcp_point_dict = gcp_reader.layer_dict['GeomaxFieldReferencePts']['feat_dict']
ref_gcp_point_names = np.array([point['ID'] for point in list(ref_gcp_point_dict.values())])

geomax_to_dgps_transform = osr.CoordinateTransformation(gcp_reader.layer_dict['GeomaxFieldReferencePts']['spatial_ref'], dgpsSrs)

field_readers = []
make_corrected_files = False
for geomax_shapefile_name in geomax_shapefile_names:
    field_reader = su.GdalVectorReader(geomax_shapefile_name)
    field_reader.read(id_field='NAME')
    # hack: srs not set, so set it here
    list(field_reader.layer_dict.values())[0]['spatial_ref'] = gcp_reader.layer_dict['GeomaxFieldReferencePts']['spatial_ref'].Clone()
    field_point_dict = OrderedDict(list(field_reader.layer_dict.values())[0]['feat_dict'])

    # find correction offset for this file (one offset that is the mean of all gcp differences)
    field_point_names = np.array(list(field_point_dict.keys()))
    gcp_idx = np.array([field_point_name.startswith('REF') for field_point_name in field_point_names])
    offsets = []
    for gcp_key in field_point_names[gcp_idx]:
        field_gcp = field_point_dict[gcp_key]
        ref_gcp = ref_gcp_point_dict[gcp_key]
        offset = [ref_gcp['X'] - field_gcp['X'], ref_gcp['Y'] - field_gcp['Y']]
        offsets.append(offset)
    # find one offset that is the mean of all gcp differences
    offset = np.array(offsets).mean(axis=0)
    print(field_point_names[gcp_idx])

    # create output corrected shapefile
    if make_corrected_files:
        corrected_shapefile_name = '{0}_Corrected.shp'.format(geomax_shapefile_name[:-4])
        ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(corrected_shapefile_name)
        # layer = ds.CreateLayer(corrected_shapefile_name[:-4], gcp_reader.layer_dict['GeomaxFieldReferencePts']['spatial_ref'].Clone(), ogr.wkbPoint)
        layer = ds.CreateLayer(corrected_shapefile_name[:-4], dgpsSrs.Clone(), ogr.wkbPoint)
        layer.CreateField(ogr.FieldDefn("ID", ogr.OFTString))

    # Add the fields we're interested in
    # apply the offset to all the points
    for key, point in field_point_dict.items():
        point['X_corr']  = point['X'] + offset[0]
        point['Y_corr'] = point['Y'] + offset[1]
        point_corr = ogr.Geometry(ogr.wkbPoint)
        point_corr.AddPoint(point['X_corr'], point['Y_corr'])
        point['geom_corr'] = point_corr
        # point_corr.Transform(geomax_to_dgps_transform)

        if make_corrected_files:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("ID", key)
            feature.SetGeometry(point_corr)
            layer.CreateFeature(feature)


    # ds = None
    field_readers.append(field_reader)

    # reformat, transform srs and add to dgpsDict
    # 'PlotName': 'TCH_03',
    # 'Comment': 'H1',
    for key, field_point in field_point_dict.items():
        if '_H' in key and not 'REF' in key :
            plot_name = key[:key.find('_H')]
            comment = key[key.find('_H')+1:]
            point = {}
            point['PlotName'] = str(plot_name).replace('_','')
            point['Comment'] = comment
            point['geom'] = field_point['geom_corr'].Clone()
            # point['geom'] = field_point['geom'].Clone()
            point['geom'].Transform(geomax_to_dgps_transform)
            point['X']= point['geom'].GetX()
            point['Y']= point['geom'].GetX()
            dgpsDict[key] = point


# field_reader.layer_dict['GEF_FIELD4']['feat_dict']['REF_Toilet_NW']['geom']

print(list(field_reader.layer_dict['GEF_FIELD4']['feat_dict'].keys()))
print(field_reader.layer_dict['GEF_FIELD4']['spatial_ref'])


## Read in available CS ground truth
from csv import DictReader

plotCsGt = {}
csGtFilenames = [r"C:\Data\Development\Projects\GEF-5 SLM\Data\Outputs\Allometry\Plot AGC.csv"]

for csGtFilename in csGtFilenames:
    with open(csGtFilename, 'r') as csGtFile:
        reader = DictReader(csGtFile)
        # print reader.fieldnames
        for row in reader:
            plotCsGt[row['ID']] = row


## Create output shapefile
# set up the shapefile driver

driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.CreateDataSource(outShapeFileName)
# osr.SpatialReference().Impo(4326)
import os
layer = ds.CreateLayer(os.path.split(outShapeFileName)[-1][:-4], dgpsSrs, ogr.wkbMultiPolygon)
# Add the fields we're interested in
# field_name = ogr.FieldDefn("Name", ogr.OFTString)
# field_name.SetWidth(64)
# ogr.OFT
layer.CreateField(ogr.FieldDefn("ID", ogr.OFTString))
layer.CreateField(ogr.FieldDefn("Stratum", ogr.OFTString))
layer.CreateField(ogr.FieldDefn("Abc", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("AbcHa", ogr.OFTReal))
# layer.CreateField(ogr.FieldDefn("YcLc", ogr.OFTReal))
# layer.CreateField(ogr.FieldDefn("YcLcHa", ogr.OFTReal))
# layer.CreateField(ogr.FieldDefn("YcUc", ogr.OFTReal))
# layer.CreateField(ogr.FieldDefn("YcUcHa", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Vol", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("VolHa", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Size", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("N", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("LitterC", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("LitterCHa", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("AgcHa", ogr.OFTReal))


plotNames = np.array([f['PlotName'] for f in list(dgpsDict.values())])

for plotName in np.unique(plotNames):
    idx = plotNames == plotName
    plotPoints = np.array(list(dgpsDict.values()))[idx]
    comments = [plotPoint['Comment'] for plotPoint in plotPoints]
    cnrIdx = [comment.startswith('H') and 'FPP' not in comment for comment in comments]
    plotCnrs = plotPoints[cnrIdx]

    print(plotName, end=' ')
    if plotName.__contains__('_') or plotName.__contains__('_'):
        sepIdx = str(plotName).find('_')
        if sepIdx < 0:
            sepIdx = str(plotName).find('-')
        lbl = plotName[:sepIdx]
        num = np.int32(plotName[sepIdx+1:])
        plotName = '%s%d'%(lbl,num)

    plotName = str(plotName).replace('_','')
    plotName = str(plotName).replace('-', '')
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("ID", plotName)
    if plotName in plotCsGt:
        # fields = ['Yc','YcHa','Size','N', 'LitterC', 'LitterCCHa', 'AgcHa', 'YcLc', 'YcLcHa', 'YcUc', 'YcUcHa', 'Vol', 'VolHa']
        # fields = ['Yc', 'YcHa', 'Size', 'N', 'LitterC', 'LitterCHa', 'AgcHa', 'Vol', 'VolHa']
        fields = ['Stratum', 'Abc', 'AbcHa', 'Size', 'N', 'LitterC', 'LitterCHa', 'AgcHa', 'Vol', 'VolHa']
        for f in fields:
            feature.SetField(f, plotCsGt[plotName][f])
    else:
        print('Agc not found for ' + plotName)
        # fields = ['Yc', 'YcHa', 'Size', 'N', 'LitterC', 'LitterCHa', 'AgcHa']
        # fields = ['Yc','YcHa','Size','N', 'LitterC', 'LitterCHa', 'AgcHa', 'YcLc', 'YcLcHa', 'YcUc', 'YcUcHa', 'Vol', 'VolHa']
        # fields = ['Yc', 'YcHa', 'Size', 'N', 'LitterC', 'LitterCHa', 'AgcHa', 'Vol', 'VolHa']
        fields = ['Stratum', 'Abc', 'AbcHa', 'Size', 'N', 'LitterC', 'LitterCHa', 'AgcHa', 'Vol', 'VolHa']
        for f in fields:
            feature.SetField(f, 0)

    # plotLinRing = ogr.Geometry(ogr.wkbLinearRing)
    # OGR / GDAL has very unintuitive behaviour with making polygon from points - the below is the best/only way I could get it done
    # Note that a polygon can have holes, so it is not simply a list of points but a collection of ring geometries which are each a list of points
    plotGeomColl = ogr.Geometry(ogr.wkbGeometryCollection)
    for plotCnr in plotCnrs:
        plotGeomColl.AddGeometry(plotCnr['geom'])
        # plotLinRing.AddPoint(plotCnr['X'], plotCnr['Y'], plotCnr['geom'].GetZ())
        print('.', end=' ')
    print()
    plotPoly = plotGeomColl.ConvexHull()
        # ogr.Geometry(ogr.wkbPolygon)
    # plotPoly.AddGeometry(plotLinRing)
    feature.SetGeometry(plotPoly)
    layer.CreateFeature(feature)
    # Dereference the feature
    # feature = None

# ds = None
ds.FlushCache()

#make csv file of points for Cos
if False:
    import os
    from csv import DictWriter
    # write out surrogate map for Cos
    outFileName = str.format('{0}\\GEF Sampling Corrected Points.csv', os.path.dirname(outShapeFileName))

    keys = ['Comment', 'PlotName', 'Horz_Prec', 'Vert_Prec', 'X', 'Y']
    with open(outFileName, 'wb') as outfile:
        writer = DictWriter(outfile, fieldnames=keys)
        writer.writeheader()
        for v in dgpsDict.values():
            vs = {k: v.get(k, None) for k in keys}
            writer.writerow(vs)

    pn = np.array([p['PlotName'] for p in dgpsDict.values()])
    np.unique(pn).size

if False:
    # Process the text file and add the attributes and features to the shapefile
    for row in reader:
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField("Name", row['Name'])
        feature.SetField("Region", row['Region'])
        feature.SetField("Latitude", row['Latitude'])
        feature.SetField("Longitude", row['Longitude'])
        feature.SetField("Elevation", row['Elev'])

        # create the WKT for the feature using Python string formatting
        wkt = "POINT(%f %f)" % (float(row['Longitude']), float(row['Latitude']))

        # Create the point from the Well Known Txt
        point = ogr.CreateGeometryFromWkt(wkt)

        # Set the feature geometry using the point
        feature.SetGeometry(point)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Dereference the feature
        feature = None
