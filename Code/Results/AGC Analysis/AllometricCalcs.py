from __future__ import print_function
from __future__ import division
from builtins import zip
from past.utils import old_div
import sys
sys.path.append("C:\Data\Development\Projects\PhD GeoInformatics\Code\Misc Tools")
import AllometryUtils as au
import SpatialUtils as su
import pylab
import numpy as np
from csv import DictWriter
import os

## updated allometric calcs with new AllometryUtils and latest field data

allometryFileName = "C:\Data\Development\Projects\PhD GeoInformatics\Data\GEF Sampling\Final Sampling March 2019\AllometricModels.xlsx"
litterFileName = "C:\Data\Development\Projects\PhD GeoInformatics\Data\GEF Sampling\Final Sampling March 2019\GEF_Litter_Consolidated ALL plots_Sent to Dugal_2019.05.29_Cos.xlsx"
woodyFileName = "C:\Data\Development\Projects\PhD GeoInformatics\Data\GEF Sampling\Final Sampling March 2019\GEF_Spp list_Consolidated_ALL plots_Sent to Dugal_2019.05.31_Cos_ProblemsSolved.xlsx"

# reload(su)
# reload(au)

allom = au.WoodyAllometryCalculator(allometry_file_name=allometryFileName, woody_file_name=woodyFileName,
                                    correction_method=au.CorrectionMethod.NicklessZou)

allom.ReadAllometryFile()
allom.ReadMasterSpeciesMap()

# compare master map and cos' map
cos_species = np.array(list(allom.cos_surrogate_map.keys()))
master_species = np.array(list(allom.master_species_map.keys()))
for species in cos_species:
    cos = allom.cos_surrogate_map[species]['allom_species']
    master = allom.master_species_map[species]['allom_species']
    if cos != master:
        print('{0} mismatch: \tCos allom: {1}\t Master allom: {2}'.format(species, cos, master))

allom.EvalAllRecordCs(make_marked_file=False)
allom.WriteAllCsFile()
# WoodyAllometryCalculator.EvalAllPlotCs(allom.plots)
allom.ReadLitter(litter_file_name=litterFileName)
allom.EvalPlotSummaryCs()
allom.WriteSummaryFile()

# write out surrogate map for Cos
outFileName = str.format('{0}\\Master Surrogate Map py3.csv', os.path.dirname(woodyFileName))

with open(outFileName, 'w') as outfile:
    writer = DictWriter(outfile, list(allom.master_species_map.values())[100].keys())
    writer.writeheader()
    writer.writerows(list(allom.master_species_map.values()))


# look at rel betw vol and yc
ycs = np.array([plot['AbcHa'] for plot in list(allom.summary_plots.values())])
agcs = np.array([plot['AgcHa'] for plot in list(allom.summary_plots.values())])
vols = np.array([plot['VolHa'] for plot in list(allom.summary_plots.values())])
ids = np.array([plot['ID'] for plot in list(allom.summary_plots.values())])
classes = np.array([plot['Stratum'] for plot in list(allom.summary_plots.values())])
litters = np.array([plot['LitterCHa'] for plot in list(allom.summary_plots.values())])

pylab.figure()
su.scatter_plot(ycs, vols,labels=ids, class_labels=classes, xlabel='YcHa', ylabel='Volume')
pylab.figure()
su.scatter_plot(agcs, vols,labels=ids, class_labels=classes, xlabel='AgcHa', ylabel='Volume')
pylab.figure()
su.scatter_plot(litters, ycs,labels=ids, class_labels=classes, xlabel='LitterCHa', ylabel='YcHa')

# ---------------------------------------------------------------------------------------------------
# plots for report
reload(su)
from sklearn import linear_model, metrics
scores, predicted = su.FeatureSelector.score_model(vols[:,None], old_div(ycs[:,None],1000), model=linear_model.LinearRegression(), find_predicted=True, cv=len(ycs), print_scores=True)

f1 = pylab.figure()
f1.set_size_inches(10, 4, forward=True)
ax = pylab.subplot(1,2,1, aspect='equal')
su.scatter_plot(old_div(vols,1000), old_div(ycs,1000), xlabel='Biomass volume ($10^3$ m$^{3}$ ha$^{-1}$)', ylabel='ABC (t C ha$^{-1}$)')
pylab.title('(a)')
# ax.set_aspect('equal')
# pylab.ticklabel_format(axis='x', style='sci', scilimits=[0,4], useOffset=True)
scores, predicted = su.FeatureSelector.score_model(vols[:,None], old_div(agcs[:,None],1000), model=linear_model.LinearRegression(), find_predicted=True, cv=len(ycs), print_scores=True)
ax = pylab.subplot(1,2,2, aspect='equal')
su.scatter_plot(old_div(vols,1000), old_div(agcs,1000), xlabel='Biomass volume ($10^3$ m$^{3}$ ha$^{-1}$)', ylabel='AGC (t C ha$^{-1}$)')
# ax.set_aspect('equal')
f1.tight_layout()
pylab.title('(b)')
f1.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\VolVsAgcScatter.png', dpi=300)


scores, predicted = su.FeatureSelector.score_model(old_div(litters[:,None],1000), old_div(ycs[:,None],1000), model=linear_model.LinearRegression(), find_predicted=True, cv=len(ycs), print_scores=True)

f1 = pylab.figure()
f1.set_size_inches(5, 4, forward=True)
su.scatter_plot(old_div(litters,1000), old_div(ycs,1000), xlabel='Litter C (t C ha$^{-1}$)', ylabel='ABC (t C ha$^{-1}$)')
pylab.tight_layout()
f1.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\LitterCVsAbcScatter.png', dpi=300)


# ---------------------------------------------------------------------------------------------------
# further data analysis (species contributions, per-stratum, plant height contributions)
import numpy as np
import pylab
import pandas as pd
from scipy.stats import gaussian_kde

plotAbcFileName = "C:\Data\Development\Projects\PhD GeoInformatics\Data\GEF Sampling\Final Sampling March 2019\GEF_Spp list_Consolidated_ALL plots_Sent to Dugal_2019.05.31_Cos_ProblemsSolved - All WoodyC.csv"
outFileName = "C:\Data\Development\Projects\PhD GeoInformatics\Data\GEF Sampling\Final Sampling March 2019\GEF_Spp list_Consolidated_ALL plots_Sent to Dugal_2019.05.31_Cos_ProblemsSolved - Per-Stratum Per-Species WoodyC.xlsx"

abcDf = pd.read_csv(plotAbcFileName)

#-------------------- per-plot per-species contributions and per-stratum per-species contributions
plotSpeciesYcDict = {}
degrSpeciesYcDict = {}
nestedPlotSize = 5.0
for id, idGroup in abcDf.groupby('ID'):
    # print '{0}: YC {1}'.format(id, idGroup['yc'].mean())
    containingPlotSize = float(idGroup['plot_size'].max())
    nestedPlotSize = 5
    isNested = np.any(idGroup['plot_size'] == nestedPlotSize)
    degrClass = idGroup['degr_class'].iloc[0].strip()
    plotSpeciesYcDict[id] = {'degr_class':degrClass, 'plot_size':containingPlotSize}
    # if degrClass not in degrSpeciesYcDict:
    #     degrSpeciesYcDict[degrClass] = {}
    for species, speciesGroup in idGroup.groupby('species'):
        nestedIdx = speciesGroup['plot_size'] == nestedPlotSize
        heightIdx = speciesGroup['height'] < 50
        nestedScaleF = (old_div(containingPlotSize,nestedPlotSize))**2
        speciesYc = 0
        speciesYc += (nestedScaleF * speciesGroup.loc[nestedIdx&heightIdx, 'yc'].sum()) + speciesGroup.loc[nestedIdx&~heightIdx, 'yc'].sum()
        speciesYc += speciesGroup.loc[~nestedIdx, 'yc'].sum()
        speciesYcHa = old_div(speciesYc * (100**2),(containingPlotSize**2))
        plotSpeciesYcDict[id][species] = speciesYc
        #
        # if species not in degrSpeciesYcDict[degrClass]:
        #     degrSpeciesYcDict[degrClass][species] = 0
        # degrSpeciesYcDict[degrClass][species] += speciesYc

plotSpeciesYcDf = pd.DataFrame(list(plotSpeciesYcDict.values()), index=list(plotSpeciesYcDict.keys()))
plotSpeciesYcDf['id'] = plotSpeciesYcDf.index
plotSpeciesYcDf = plotSpeciesYcDf.fillna(value=0)

# degrSpeciesYcDf = pd.DataFrame(degrSpeciesYcDict.values(), index=degrSpeciesYcDict.keys())
# degrSpeciesYcDf['degr_class'] = degrSpeciesYcDf.index
# degrSpeciesYcDf = degrSpeciesYcDf.fillna(value=0)
# degrSpeciesYcDf.to_excel(outFileName)
#
#
# speciesKeys = plotSpeciesYcDf.keys().drop(['degr_class', 'plot_size', 'id'])
# print degrSpeciesYcDf.groupby('degr_class')[speciesKeys].sum()

#-------------------- make plots of 10 highest contributing species per stratum
degrSpeciesYcDf = pd.DataFrame()
overallDegrSpeciesYcDf = pd.DataFrame()
overallPlotAreaTtl = np.array([ps ** 2 for ps in plotSpeciesYcDf['plot_size']]).sum()

f1 = pylab.figure()
f1.set_size_inches(10, 4, forward=True)
plotIdxs = [1, 3, 2]
for (degrClass, degrGroup), plotIdx in zip(plotSpeciesYcDf.groupby('degr_class', sort=False), plotIdxs):
    speciesKeys = degrGroup.keys().drop(['degr_class', 'plot_size', 'id'])
    speciesYcTtl = degrGroup[speciesKeys].sum()
    plotAreaTtl = np.array([ps**2 for ps in degrGroup['plot_size']]).sum()
    speciesYcHa = old_div((100**2)*speciesYcTtl,(1000*plotAreaTtl))      # tC/ha

    degrRow = dict(list(zip(speciesKeys, speciesYcHa)))
    degrRow['degr_class'] = degrClass
    degrSpeciesYcDf = degrSpeciesYcDf.append(degrRow, ignore_index=True)

    if False:
        degrRow = dict(list(zip(speciesKeys, old_div((100**2)*speciesYcTtl,(1000*overallPlotAreaTtl)))))
        degrRow['degr_class'] = degrClass
        overallDegrSpeciesYcDf = overallDegrSpeciesYcDf.append(degrRow, ignore_index=True)

    print(speciesYcHa.sum())
    idx = np.argsort(-speciesYcHa)
    pylab.subplot(1, 3, plotIdx)
    pylab.bar(np.arange(0, 10), speciesYcHa[idx[:10]], label=degrClass)
    pylab.xticks(np.arange(0, 10), speciesKeys[idx[:10]], rotation='vertical')  # prop={'size':fontSize-2})
    pylab.title(degrClass)
    pylab.ylabel('t C ha$^{-1}$')
    pylab.tight_layout()

f1.savefig(r'C:\Data\Development\Projects\PhD GeoInformatics\Docs\Funding\GEF5\Invoices, Timesheets and Reports\Final Report\SpeciesAbcPerStratum.png', dpi=300)
degrSpeciesYcDf.to_excel(outFileName)

#-------------------- make plot of ABC contribution by height
abcDf.loc[abcDf['degr_class']=='Degraded ', 'degr_class'] = 'Degraded'
nestedIdx = abcDf['plot_size']==5
plantHeights = abcDf.loc[nestedIdx, 'height']

kde = gaussian_kde(plantHeights)  # , bw_method=bandwidth / height.std(ddof=1))
heightGrid = np.linspace(0, 500, 200)
heightKde = kde.evaluate(heightGrid)
pylab.figure()
pylab.subplot(2,2,4)
pylab.plot(heightGrid, heightKde)
axLim = pylab.axis()
h = pylab.plot([50, 50], [0, heightKde.max()], 'r')
pylab.xlabel('Plant Height (cm)')
pylab.ylabel('Probability')
pylab.title('Overall')

plotIdxs = [1, 3, 2]
for (degrClass, degrGroup), plotIdx in zip(abcDf.groupby('degr_class'), plotIdxs):
    nestedIdx = degrGroup['plot_size'] == 5
    plantHeights = degrGroup.loc[nestedIdx, 'height']

    kde = gaussian_kde(plantHeights)  # , bw_method=bandwidth / height.std(ddof=1))
    heightGrid = np.linspace(0, 500, 200)
    heightKde = kde.evaluate(heightGrid)
    pylab.subplot(2, 2, plotIdx)
    pylab.plot(heightGrid, heightKde)
    axLim = pylab.axis()
    h = pylab.plot([50, 50], [0, heightKde.max()], 'r')
    pylab.xlabel('Plant Height (cm)')
    pylab.ylabel('Probability')
    pylab.title(degrClass)
    pylab.tight_layout()

nestedAbcDf = abcDf.loc[abcDf['plot_size']==5]
nestedAbcDf = nestedAbcDf.sort_values(by='height')
abcCumSum = np.cumsum(nestedAbcDf['yc'])

pylab.figure()
pylab.subplot(2, 2, 4)
pylab.plot(nestedAbcDf['height'], old_div((100*abcCumSum),abcCumSum.max()))
axLim = pylab.axis()
h = pylab.plot([50, 50], [axLim[2], axLim[3]], 'r')
pylab.axis(axLim)
pylab.grid('on')
pylab.xlabel('Plant Height (cm)')
pylab.ylabel('% of ABC')
pylab.title('Overall')

plotIdxs = [1, 3, 2]
for (degrClass, degrGroup), plotIdx in zip(nestedAbcDf.groupby('degr_class'), plotIdxs):
    degrGroup = degrGroup.sort_values(by='height')
    abcCumSum = np.cumsum(degrGroup['yc'])

    pylab.subplot(2, 2, plotIdx)
    pylab.plot(degrGroup['height'], old_div((100 * abcCumSum), abcCumSum.max()))
    axLim = pylab.axis()
    h = pylab.plot([50, 50], [axLim[2], axLim[3]], 'r')
    pylab.axis(axLim)
    pylab.grid('on')
    pylab.xlabel('Plant Height (cm)')
    pylab.ylabel('% of ABC')
    pylab.title(degrClass)
    pylab.tight_layout()

# TODO  - we can simulate what the "error" is when we increase the height cutoff idx (exclude heights less than x in
#  containing plot, and extrap heights < x from nested plot, then compare to x=50)

if False:
    #-------------------- make plots of 10 highest contributing species overall
    overallSpeciesKeys = list(plotSpeciesYcDf.keys()).drop(['degr_class', 'plot_size', 'id'])
    overallSpeciesYcTtl = plotSpeciesYcDf[overallSpeciesKeys].sum()
    overAllSpeciesYcHa = old_div((100 ** 2) * overallSpeciesYcTtl, (1000 * overallPlotAreaTtl))         # tC/ha
    overAllSpeciesIdx = np.argsort(-overAllSpeciesYcHa)

    degrRow = dict(list(zip(overallSpeciesKeys, overAllSpeciesYcHa)))
    degrRow['degr_class'] = 'Overall'
    # overallDegrSpeciesYcDf = overallDegrSpeciesYcDf.append(degrRow, ignore_index=True)

    f = pylab.figure()
    f.set_size_inches(10, 4, forward=True)
    bottom = np.zeros(10)
    x = np.arange(0., 10.)
    reload(pd)
    reload(np)
    reload(pylab)
    colours = ['red','green','orange']
    for i in [0,2,1]:
        degrRow = overallDegrSpeciesYcDf.iloc[i]
        y = degrRow[speciesKeys[overAllSpeciesIdx[:10]]].values.astype('float')
        np.isfinite(y)
        pylab.isfinite(y)
        # pd.isfinite(y)
        pylab.bar(x, y, bottom=bottom, label=degrRow['degr_class'], facecolor=colours[i])
        bottom += y.copy()
        print(degrRow['degr_class'])

    pylab.xticks(np.arange(0, 10), overallSpeciesKeys[overAllSpeciesIdx[:10]], rotation='vertical')  # prop={'size':fontSize-2})
    pylab.tight_layout()
    pylab.ylabel('t C ha$^{-1}$')
    pylab.legend()

# ------------------check per startum yc with summary file
if False:
    summaryAbcFileName = "C:\Data\Development\Projects\PhD GeoInformatics\Data\GEF Sampling\Final Sampling March 2019\GEF_Spp list_Consolidated_ALL plots_Sent to Dugal_2019.05.31_Cos_ProblemsSolved - Summary Woody & Litter.csv"
    summaryAbcDf = pd.read_csv(summaryAbcFileName)
    summaryAbcDf.loc[summaryAbcDf['Degr. Class']=='Degraded ', 'Degr. Class'] = 'Degraded'

    print(summaryAbcDf.groupby('Degr. Class')['YcHa'].mean())
    print(summaryAbcDf.groupby('Degr. Class')['AgbHa'].mean())
