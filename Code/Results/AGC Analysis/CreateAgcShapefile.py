#
   """
      GEF5-SLM Above ground carbon estimation in thicket using multi-spectral images
      Copyright (C) 2020 Dugal Harris
      Released under GNU Affero General Public License (AGPL) (https://www.gnu.org/licenses/agpl.html)
      email dugalh@gmail.com
   """




#

      GEF5-SLM Above ground carbon estimation in thicket using multi-spectral images
      Copyright (C) 2020  Dugal Harris

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU Affero General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU Affero General Public License for more details.

      You should have received a copy of the GNU Affero General Public License
      along with this program.  If not, see <https://www.gnu.org/licenses/>.



from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
import gdal
import ogr
import numpy as np
import osr
from collections import OrderedDict
# import pylab
# from scipy import stats as stats
# from matplotlib import patches
# import matplotlib.pyplot as plt
# from scipy import ndimage as ndimage
#
# # Python Imaging Library imports
# from PIL import Image
# from PIL import ImageDraw
# sys.path.append("C:\Data\Development\Projects\PhD GeoInformatics\Code\Misc Tools")
from modules import modelling as su

# |layerid=0|subset="Comment" LIKE 'H%'
correctedShapeFileNames = [
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\Sept 2017\Point_ge.shp",
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\Dec 2017\Point_ge.shp",
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\March 2018\Point_ge.shp",
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\June 2018\Point_ge.shp",
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\Sept 2018\Point_ge.shp",
    ]

outShapeFileName = r"C:\Data\Development\Projects\GEF-5 SLM\Data\Outputs\Geospatial\GEF Plot Polygons with AGC.shp"

def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = ((x - ulX)/xDist)
    line = ((y - ulY)/yDist)
    return (pixel, line)


# for correctedShapeFileNames in correctedShapeFileNames:
#     ds = su.GdalVectorReader(correctedShapeFileNames)
#     data = ds.read()

## read dgps points into dictionary
dgpsSrs = osr.SpatialReference()
dgpsDict = OrderedDict()
for correctedShapeFileName in correctedShapeFileNames:
    # ds = gdal.OpenEx(correctedShapeFileName, gdal.OF_VECTOR)
    ds = gdal.OpenEx(correctedShapeFileName, gdal.OF_VECTOR)
    if ds is None:
        print('Failed to open {0}'.format(correctedShapeFileName))
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
geomax_shapefile_names = [
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\March 2019\GEF_FIELD1.shp",
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\March 2019\GEF_FIELD2.shp",
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\March 2019\GEF_FIELD3.shp",
    r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\March 2019\GEF_FIELD4.shp",
    ]

# the ref file for adjusting the above
gcp_shapefile_name = r"C:\Data\Development\Projects\GEF-5 SLM\Data\Sampling Inputs\Plot locations\March 2019\GeomaxFieldReferencePts.shp"

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


## create output shapefile
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

        # create the point from the Well Known Txt
        point = ogr.CreateGeometryFromWkt(wkt)

        # Set the feature geometry using the point
        feature.SetGeometry(point)
        # create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Dereference the feature
        feature = None
