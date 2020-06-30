'''

'''
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import object
from past.utils import old_div
from openpyxl import load_workbook
import numpy as np
import pylab
from scipy.stats import gaussian_kde
import collections
from csv import DictWriter
from collections import OrderedDict
import os.path
from scipy import stats as stats
from enum import Enum
import warnings

fontSize = 16


def FormatSpecies(species):
    ''' Formats the species name into abbreviated dot notation

    Parameters
    ----------
    species
        the species name e.g. 'Portulacaria afra'

    Returns
    -------
        the abbreviated species name e.g. 'P.afra'        
    '''
    species = str(species).strip()
    comps = species.split(' ', 1)
    if len(comps) > 1:
        abbrev_species = str.format(str("{0}.{1}"), comps[0][0], comps[1].strip())
    else:
        abbrev_species = species
    return abbrev_species

class CorrectionMethod(Enum):
    ''' Aboveground biomass correction method
    '''
    Duan = 1
    MinimumBias = 2
    NicklessZou = 3

class AbcModel(object):
    def __init__(self, model_dict={}, surrogate_dict=None, correction_method=CorrectionMethod.NicklessZou):
        ''' Estimate woody / above ground carbon (ABC) from allometric measurements

        Parameters
        ----------
        model_dict
            dictionary of woody allometric models
            (see table 3 in https://www.researchgate.net/publication/335531470_Aboveground_biomass_and_carbon_pool_estimates_of_Portulacaria_afra_spekboom-rich_subtropical_thicket_with_species-specific_allometric_models)
        surrogate_dict
            dictionary of known surrogate species for which models exist
            surrogate species are species contained in the keys of model_dict i.e. for which models exist
            (see supplementary data in https://doi.org/10.1016/j.foreco.2019.05.048)
        correction_method
            ABC correction method to use (default CorrectionMethod.NicklessZou)
        '''
        self.model_dict = model_dict
        self.surrogate_dict = surrogate_dict
        self.correction_method = correction_method

    def EstimateAbc(self, meas_dict={'canopy_length': 0., 'canopy_width': 0., 'height': 0.}):
        ''' Apply allometric model to measurements

        Parameters
        ----------
        meas_dict
            dictionary of plant measurements

        Returns
        -------
            ABC results in a dict (see fields in supplementary data in https://doi.org/10.1016/j.foreco.2019.05.048)
        '''
        res = {'yc':0.,     # Above ground carbon (ABC) in kg
               'yc_lc':0.,  #
               'yc_uc':0.,
               'height':0.,
               'area':0.,
               'vol':0.}
        species =str(meas_dict['species']).strip()

        x = 0.
        CD = np.mean([meas_dict['canopy_length'], meas_dict['canopy_width']])
        CA = np.pi * (CD/2)**2
        # NB leave height in cm as other code assumes cm, but convert vol and area to m to avoid large num + overflow
        res['height'] = meas_dict['height']                 # pass through plant height (cm)
        res['area'] = CA/1.e4                               # canopy area (m**2)
        res['vol'] = CA*meas_dict['height']/1.e6            # cylindrical plant volume in (m**3)

        # check in surrogate_dict , then return area / vol only (yc = 0)
        if species not in self.surrogate_dict:
            warnings.warn('{0} has no key in species map, setting yc = 0'.format(species))
            return res
        allom_species = self.surrogate_dict[species]['allom_species']
        if allom_species == 'none':
            warnings.warn('{0} has no model, setting yc = 0'.format(species))
            return res
        elif allom_species not in self.model_dict:  # should never happen
            raise Exception('{0} has no key in model_dict'.format(allom_species))

        model = self.model_dict[allom_species]

        if model['vars'] == 'CA.H':
            x = CA*meas_dict['height']
        elif model['vars'] == 'CA.SL':
            x = CA*meas_dict['height']
        elif model['vars'] == 'CD':
            x = CD
        elif model['vars'] == 'CD.H':
            x = CD*meas_dict['height']
        elif model['vars'] == 'Hgt':
            x = meas_dict['height']
        else:
            raise Exception('{0}: unknown variable', model['vars'])

        yn = np.exp(model['ay'])*x**model['by']   # "naive"

        # correct to yc
        if self.correction_method == CorrectionMethod.Duan:
            yc = yn * model['Duan']
        elif self.correction_method == CorrectionMethod.MinimumBias:
            yc = yn * model['MB']
        else:                     # CorrectionMethod.NicklessZou
            if yn == 0.:
                yc = 0.
            else:
                yc = np.exp(np.log(yn) + (model['sigma']**2)/2.)

        if model['use_wd_ratio']:
            wd_species = self.master_species_map[species]['wd_species']
            if wd_species not in self.wd_ratios:
                print('WARNING: Evalmeas_dictCs - {0} has no key in wd_ratios, using 1'.format(wd_species))
            else:
                yc = yc * self.wd_ratios[wd_species]['wd_ratio']

        res['yc'] = yc * 0.48
        res['yc_lc'] = yc * model['LC']
        res['yc_uc'] = yc * model['UC']
        return res

## Class to read in allom and wd ratio tables, and calc yc for a record
class AgcAllometry(object):
    '''
    '''
    def __init__(self, model_file_name="", data_file_name="", correction_method = CorrectionMethod.NicklessZou):
        '''

        Parameters
        ----------
        model_file_name
        data_file_name
        correction_method
        '''
        self.allometry_file_name = model_file_name
        self.woody_file_name = data_file_name
        if not os.path.exists(model_file_name):
            raise Exception("Allometry file does not exist: {0}".format(model_file_name))
        if not os.path.exists(data_file_name):
            raise Exception("WoodyC file does not exist: {0}".format(data_file_name))
        self.allom_models = {}
        self.wd_ratios = {}
        self.mvdv_surrogate_map = {}
        self.master_species_map = {}
        self.plots = {}
        self.litter_dict = {}
        self.correction_method = correction_method
        # self.ReadAllometryFile()
        # self.ReadMasterSpeciesMap()


    def ReadAllometryFile(self):
        wb = load_workbook(self.allometry_file_name)
        # read in a dictionary of allometric models
        ws = wb["Allometric Models"]
        first_row = ws[1]
        header = []
        for c in first_row:
            header.append(c.value)

        self.allom_models = {}
        for r in ws[2:ws.max_row]:  # how to find num rows?
            if r[0].value is None:
                break
            # species = str.strip(str(r[0].value[0]) + '.' + str(r[1].value))
            species = str.strip(str(r[0].value)) + ' ' + str.strip(str(r[1].value))
            species = FormatSpecies(species)
            model = {}
            # model['allom_species'] = species
            model['vars'] = r[3].value
            model['ay'] = r[6].value
            model['by'] = r[7].value
            model['sigma'] = r[8].value
            model['LC'] = r[9].value
            model['UC'] = r[10].value
            model['Duan'] = r[12].value
            model['MB'] = r[13].value
            if str(r[14].value) == 'x':
                model['use_wd_ratio'] = False
            else:
                model['use_wd_ratio'] = True
            self.allom_models[species] = model
        ws = None

        # read in a dictionary of wet/dry ratios
        ws = wb["Wet Dry Ratios"]
        first_row = ws[0 + 1]
        header = []
        for c in first_row:
            header.append(c.value)

        self.wd_ratios = {}
        for r in ws[2:ws.max_row]:  # how to find num rows?
            if r[0].value is None:
                break
            # species = str.strip(str(r[0].value[0]) + '.' + str(r[1].value))
            species = str.strip(str(r[0].value)) + ' ' + str.strip(str(r[1].value))
            species = FormatSpecies(species)
            model = {}
            model['wd_species'] = species
            model['wd_ratio'] = r[4].value
            self.wd_ratios[species] = model
        ws = None

        # read in a Marius' surrogate table
        ws = wb["Surrogates"]
        first_row = ws[0 + 1]
        header = []
        for c in first_row:
            header.append(c.value)

        self.mvdv_surrogate_map = {}
        for r in ws[2:ws.max_row]:
            if r[0].value is None:
                break

            species = str.strip(str(r[0].value))
            species = FormatSpecies(species)
            model = {}
            model['species'] = species
            model['allom_species'] = FormatSpecies(str.strip(str(r[1].value)))
            if r[2].value is not None:
                model['wd_species'] = FormatSpecies(str.strip(str(r[2].value)))
            else:
                model['wd_species'] = None
            self.mvdv_surrogate_map[species] = model
        wb.close()
        wb = None


    def ReadMasterSpeciesMap(self):
        import copy
        wb = load_workbook(self.woody_file_name)

        self.cos_surrogate_map = {}
        ws = wb.get_sheet_by_name("Master spp list")
        first_row = ws[1]
        header = []
        for c in first_row:
            header.append(c.value)
        for r in ws[2:ws.max_row]:
            # if r[2].value is None:  # no mapping yet
            #     continue
            species = str(r[1].value).strip()

            for c in r[6:]:     # check this - Cos changed the spreadsheet format
                if c.value is not None and c.value != "":
                    map_species = str(c.value).strip()  # .replace('. ', '.')
                    break
            # map_species = str(r[3].value).strip().replace('. ', '.')
            model = {}
            model['species'] = species
            model['allom_species'] = FormatSpecies(map_species)
            if species in self.wd_ratios:
                model['wd_species'] = species
            elif model['allom_species'] in self.wd_ratios:
                model['wd_species'] = model['allom_species']
            else:
                model['wd_species'] = None

            model['species_full'] = str(r[0].value).strip()  # hack for translating between name formats
            self.cos_surrogate_map[species] = model
        #     if not self.allom_models.has_key(map_species):
        #         print self.cos_surrogate_map[species], " not found in allometric models"
        #         c.fill = PatternFill(fgColor='FFEE08', fill_type='solid')
        #     else:
        #         c.fill = PatternFill(fgColor='FFFFFF', fill_type='solid')
        # wb.save(filename=woodyErrorFileName)
        wb.close()
        wb = None
        if False:   # try use only Cos' map
            self.master_species_map = self.cos_surrogate_map
            return

        # combine the two species maps into one
        self.master_species_map = {}
        # first copy across Marius map as is
        print('Parse MVDV surrogate map---------------------------------------------------------------------------------')
        for species, map in self.mvdv_surrogate_map.items():
            # use abbrev name as the key always
            self.master_species_map[species] = copy.deepcopy(map)
            # self.master_species_map[species]['species'] = species
            # error check this map
            if map['allom_species'] not in self.allom_models:
                print('WARNING: species: {0}, allom surrogate: {1} - No allometric model'.format(species, map['allom_species']))
            else:
                if self.allom_models[map['allom_species']]['use_wd_ratio']:
                    if map['wd_species'] is not None:
                        if map['wd_species'] not in self.wd_ratios:
                            print('WARNING: species: {0}, wd surrogate: {1} - No wet:dry ratio'.format(species, map['wd_species']))
                            self.master_species_map[species]['wd_species'] = None
                    else:
                        print('WARNING: species: {0}, no wd surrogate'.format(species))
                else:
                    print('NOTE: species: {0}, allom surrogate {1}, needs no wet:dry surrogate'.format(species, map['allom_species']))
                    # self.master_species_map[species]['wd_species'] = None

        # now add Cos' map, adding wd_ratios
        print('Parse CB master map-------------------------------------------------------------------------------------')
        for species, map in self.cos_surrogate_map.items():
            if species not in self.master_species_map:
                self.master_species_map[species] = copy.deepcopy(map)
                # self.master_species_map[species]['species'] = species

                if map['allom_species'] == 'none':  # this has no surrogate at all and is excluded
                    self.master_species_map[species]['wd_species'] = None
                    print('NOTE: species: {0}, has no surrogates '.format(species))
                # if the source species has wd_ratio then refer to this directly
                elif species in self.wd_ratios:
                    self.master_species_map[species]['wd_species'] = species
                # else if the allom species has wd_ratio then refer to this
                elif map['allom_species'] in self.wd_ratios:
                    self.master_species_map[species]['wd_species'] = map['allom_species']
                # else check we actually need a wd ratio
                elif self.allom_models[map['allom_species']]['use_wd_ratio']:
                    # look ip wd species from in surrogate map for allom surrogate if possible
                    if map['allom_species'] in self.mvdv_surrogate_map and self.mvdv_surrogate_map[map['allom_species']]['wd_species'] is not None:
                        self.master_species_map[species]['wd_species'] = self.mvdv_surrogate_map[map['allom_species']]['wd_species']
                    else:
                        print('WARNING: species: {0}, allom surrogate: {1} - No wd surrogate'.format(species, map['allom_species']))
                        # self.master_species_map[species]['wd_species'] = ''
                else:
                    print('NOTE: species: {0}, allom surrogate {1}, needs no wet:dry surrogate'.format(species, map['allom_species']))
            else:
                print('NOTE: CB species: {0}, already exists in the MvdV map'.format(species, species))

    def EvalRecordCs(self, record, correction_method = CorrectionMethod.Duan):
        # vars = [model['vars'] for model in allometricModels.values()]
        res = {'yc':0.,'yc_lc':0.,'yc_uc':0., 'height':0., 'area':0., 'vol':0.}
        species =str(record['species']).strip()

        x = 0.
        CD = np.mean([record['canopy_length'], record['canopy_width']])
        CA = np.pi * (old_div(CD,2))**2
        # NB leave height in cm as other code assumes cm, but convert vol and area to m to avoid large num + overflow
        res['height'] = record['height']           # cm
        res['area'] = old_div(CA, 1.e4)                     # m**2
        res['vol'] = old_div(CA*record['height'], 1.e6)     # volume of a cylinder (m**3)
        # res['vol'] = (np.pi*4./3) * record['canopyLength'] * record['canopyWidth'] * record['height']/2.   # volume of an ellipse

        # error check and if there is no allom model, then return area / vol only (yc = 0)
        if species not in self.master_species_map:
            print('WARNING: EvalRecordCs - {0} has no key in species map, setting yc = 0'.format(species))
            return res
        allom_species = self.master_species_map[species]['allom_species']
        if allom_species == 'none':
            print('NOTE: EvalRecordCs - {0} has no model, setting yc = 0'.format(species))
            return res
        elif allom_species not in self.allom_models:  # should never happen
            raise Exception('WARNING: EvalRecordCs - {0} has no key in alommetry models, setting yc = 0'.format(allom_species))
            # return res
        model = self.allom_models[allom_species]

        if model['vars'] == 'CA.H':
            x = CA*record['height']
        elif model['vars'] == 'CA.SL':
            x = CA*record['height']
        elif model['vars'] == 'CD':
            x = CD
        elif model['vars'] == 'CD.H':
            x = CD*record['height']
        elif model['vars'] == 'Hgt':
            x = record['height']
        else:
            raise Exception('{0}: unknown variable', model['vars'])

        yn = np.exp(model['ay'])*x**model['by']   # "naive"

        # correct to yc
        if correction_method == CorrectionMethod.Duan:
            yc = yn * model['Duan']
        elif correction_method == CorrectionMethod.MinimumBias:
            yc = yn * model['MB']
        else:                     # CorrectionMethod.NicklessZou
            if yn == 0.:
                yc = 0.
            else:
                yc = np.exp(np.log(yn) + (model['sigma']**2)/2.)

        if model['use_wd_ratio']:
            wd_species = self.master_species_map[species]['wd_species']
            if wd_species not in self.wd_ratios:
                print('WARNING: EvalRecordCs - {0} has no key in wd_ratios, using 1'.format(wd_species))
            else:
                yc = yc * self.wd_ratios[wd_species]['wd_ratio']

        res['yc'] = yc * 0.48
        res['yc_lc'] = yc * model['LC']
        res['yc_uc'] = yc * model['UC']
        return res

    def ReadLitter(self, litter_file_name=None):
        self.litter_dict = {}
        if litter_file_name is None:
            litter_file_name = self.woody_file_name
        wb = load_workbook(litter_file_name, data_only=True)

        ws = wb['Final Litter_copied']
        for r in ws[2:ws.max_row]:
            id = str(r[0].value).strip().upper()
            id = id.replace('-0', '')
            id = id.replace('-', '')
            if id == '' or id is None or id == 'NONE' or r[1].value == 0:
                print('WARNING: ReadLitter - No ID, continue')
                continue
            # else:
            #     print 'ReadLitter - ID: {0}'.format(id)
            if not np.isreal(r[1].value) or r[1].value is None:
                dry_weight = 0.
                print('WARNING: ReadLitter - no data for {0}'.format(id))        # these are typically excluded / not sampled plots
            else:
                dry_weight = r[1].value

            if id in self.litter_dict:
                self.litter_dict[id]['dry_weight'] += dry_weight
                print('WARNING: ReadLitter - multiple values for {0}'.format(id))
            else:
                self.litter_dict[id] = {'dry_weight': dry_weight}
        wb.close()
        wb = None

    def EvalAllRecordCs(self, woody_file_name='', make_marked_file=False):
        from openpyxl.styles import PatternFill
        from openpyxl.styles.colors import Color
        from openpyxl.styles import colors
        ok_colour = Color(auto=True)

        if woody_file_name == '':
            woody_file_name = self.woody_file_name
        wb = load_workbook(self.woody_file_name)
        ws = wb.get_sheet_by_name("Consolidated data")

        print(ws.title, ' rows: ', ws.max_row)
        first_row = ws[1]
        header = []
        for c in first_row:
            header.append(c.value)

        self.unmodelled_species = {'unknown': {}, 'none': {}}

        self.plots = {}
        plots = collections.OrderedDict()
        for r in ws[2:ws.max_row]:
            if r is None or r[2].value is None:
                continue
            # print  r[2].value
            record = OrderedDict()
            species = str(r[3].value).strip()

            id = str(r[0].value).strip()
            dashLoc = str(id).find('-')
            if dashLoc < 0:
                dashLoc = 2
            id = id.replace('-', '').upper()
            idNum = np.int32(id[dashLoc:])  # get rid of leading zeros
            id = '%s%d' % (id[:dashLoc], idNum)
            plot_size = np.int32(str(r[1].value).lower().split('x')[0])

            record['ID'] = id
            record['degr_class'] = str(r[2].value)
            record['orig_species'] = species
            record['canopy_width'] = r[4].value
            record['canopy_length'] = r[5].value
            record['height'] = r[6].value
            record['species'] = species
            record['plot_size'] = plot_size

            # zero empty records
            fields = ['canopy_width', 'canopy_length', 'height']
            fields_ok = True
            for f in fields:
                if record[f] is None:
                    print('WARNING: EvalAllCs = ID: {0}, species: {1}, has incomplete data'.format(id, species))
                    record[f] = 0
                    fields_ok = False

            if r.__len__() > 7:
                record['bsd'] = str(r[7].value)
            else:
                record['bsd'] = ""

            res = self.EvalRecordCs(record, correction_method=self.correction_method)
            for key in list(res.keys()):
                record[key] = res[key]
            # record['yc'] = res['yc']
            # record['area'] = res['area']
            # record['vol'] = res['vol']

            if make_marked_file:    #mark problem cells
                if species not in self.master_species_map or not fields_ok:
                    r[3].fill = PatternFill(fgColor=colors.YELLOW, fill_type='solid')
                    print('Marking row {0}'.format(r[3].row))
                else:
                    r[3].fill = PatternFill(fgColor=ok_colour, fill_type='solid',)

            # we don't need to error check here - it is handled in EvalRecordCs, this is just to remember unknown species
            if species not in self.master_species_map or self.master_species_map[species]['allom_species'] == 'none':
                if species not in self.master_species_map:
                    key = 'unknown'
                elif self.master_species_map[species]['allom_species'] == 'none':
                    key = 'none'
                # print 'WARNING: EvalRecordCs - {0} has no key in species map, setting yc = 0'.format(species)
                # return res
                if species in self.unmodelled_species:
                    self.unmodelled_species[key][species]['count'] += 1
                    self.unmodelled_species[key][species]['vol'] += old_div(res['vol'],1.e6)
                else:
                    self.unmodelled_species[key][species] = {}
                    self.unmodelled_species[key][species]['count'] = 1
                    self.unmodelled_species[key][species]['vol'] = old_div(res['vol'], 1.e6)
            # if not self.master_species_map.has_key(species):
            #     print 'WARNING: EvalAllCs = {0} not found in species map, ommitting'.format(species)
            #     continue
            # print record

            if id in plots:
                plots[id].append(record)
            else:
                plots[id] = [record]

        if make_marked_file:
            out_file_name = str.format('{0}/{1}_Marked.xlsx', os.path.dirname(self.woody_file_name),
                                       os.path.splitext(os.path.basename(self.woody_file_name))[0])
            wb.save(filename=out_file_name)

        wb.close()
        wb = None
        self.plots = plots
        print('Unknown species:')
        for k, v in self.unmodelled_species['unknown'].items():
            print(k, v)
        print('Unmodelled species:')
        for k, v in self.unmodelled_species['none'].items():
            print(k, v)

    # @staticmethod
    def EvalPlotSummaryCs(self):
        if len(self.plots) == 0:
            print('EvalAllRecordCs has not been called')
            return
        self.summary_plots = {}
        # summing yc_lc and yc_uc is probably not valid - summing would average out errors and reduce lc and uc
        # fields_to_summarise = ['yc', 'yc_lc', 'yc_uc', 'height', 'vol']
        fields_to_summarise = ['yc', 'height', 'vol']
        for id, plot in self.plots.items():
            plot_sizes = np.array([record['plot_size'] for record in plot])
            plot_sizes_un = np.unique(plot_sizes)
            vectors_to_summarise = {}
            vector_dict = {}
            for field in fields_to_summarise:
                vectors_to_summarise[field] = np.array([record[field] for record in plot])
                vector_dict[field] = {'v': vectors_to_summarise[field]}

            heights = np.array([record['height'] for record in plot])
            # ycs = np.array([record['yc'] for record in plot])
            # yc_lcs = np.array([record['yc_lc'] for record in plot])
            # yc_ucs = np.array([record['yc_uc'] for record in plot])
            # vols = np.array([record['vol'] for record in plot])
            # vector_dict = {'yc': {'v': ycs}, 'yc_lc': {'v': yc_lcs},'yc_uc': {'v': yc_ucs}, 'vol': {'v': vols}, 'height': {'v': heights}}
            if plot_sizes_un.size > 1:      # it is a nested plot, so we need to extrapolate
                nest_record_idx = plot_sizes == plot_sizes_un.min()
                small_record_idx = heights < 50

                for k, v in vector_dict.items():
                    vv = v['v']
                    nest_v = vv[nest_record_idx & ~small_record_idx]
                    small_v = vv[nest_record_idx & small_record_idx]
                    out_v = vv[~nest_record_idx]
                    sum_v = out_v.sum() + nest_v.sum() + (small_v.sum() * ((old_div(plot_sizes_un.max(), plot_sizes_un.min())) ** 2))
                    mean_v = out_v.tolist() + nest_v.tolist() + (small_v.tolist() * ((old_div(plot_sizes_un.max(), plot_sizes_un.min())) ** 2))
                    v['sum_v'] = sum_v
                    v['mean_v'] = np.array(mean_v).mean()
                    v['n'] = out_v.size + nest_v.size + (small_v.size * (old_div(plot_sizes_un.max(), plot_sizes_un.min())) ** 2)
            else:
                for k, v in vector_dict.items():
                    vv = v['v']
                    v['sum_v'] = vv.sum()
                    v['mean_v'] = vv.mean()
                    v['n'] = vv.size

            summary_plot = OrderedDict()
            summary_plot['ID'] = id
            summary_plot['Stratum'] = plot[0]['degr_class']
            summary_plot['Size'] = plot_sizes_un.max()
            summary_plot['N'] = vector_dict['height']['n']
            summary_plot['Vol'] = vector_dict['vol']['sum_v']
            summary_plot['VolHa'] = old_div((100. ** 2) * vector_dict['vol']['sum_v'], (plot_sizes_un.max() ** 2))
            summary_plot['Height'] = vector_dict['height']['sum_v']
            summary_plot['HeightHa'] = old_div((100. ** 2) * vector_dict['height']['sum_v'], (plot_sizes_un.max() ** 2))
            summary_plot['MeanHeight'] = vector_dict['height']['mean_v']
            summary_plot['Abc'] = vector_dict['yc']['sum_v']
            summary_plot['AbcHa'] = old_div((100. ** 2) * vector_dict['yc']['sum_v'], (plot_sizes_un.max() ** 2))
            # summary_plot['AbcLc'] = vector_dict['yc_lc']['sum_v']
            # summary_plot['AbcLcHa'] = (100. ** 2) * vector_dict['yc_lc']['sum_v'] / (plot_sizes_un.max() ** 2)
            # summary_plot['AbcUc'] = vector_dict['yc_uc']['sum_v']
            # summary_plot['AbcUcHa'] = (100. ** 2) * vector_dict['yc_uc']['sum_v'] / (plot_sizes_un.max() ** 2)

            if len(self.litter_dict) == 0:
                print('WARNING: EvalPlotSummaryCs - no litter data')
            else:
                if id in self.litter_dict and self.litter_dict[id]['dry_weight'] > 0.:
                    summary_plot['LitterC'] = old_div(0.48 * self.litter_dict[id]['dry_weight'], 1000)  # g to kg
                    # the litter quadrats are uniform across all plot sizes.  the dry weight is converted to carbon using the factor 0.48 (according to Cos)
                    # or 0.37 (according to James / CDM)
                    # current the trial plots have no litter data (INT* and TCH*)
                    # NB note, there is a factor of 44/12 in the CDM equations that I don't quite understand.  I think it is a conversion from C weight to
                    # CO2 weight (i.e. a chemical thing that represents not how much carbon is stored by how much C02 was captured out the atmosphere).
                    # we should make sure that we are using the same units in the woody and litter C i.e. we need to check what units Marius eq give
                    # Mills and van der Vyver use .48
                    summary_plot['LitterCHa'] = old_div(summary_plot['LitterC'] * (100. ** 2), (4 * (0.5 ** 2)))
                    # summary_plot['LitterCHa'] = summary_plot['LitterC'] * 0.37 * (100. ** 2) / (4 * (0.5 ** 2))
                    summary_plot['AgcHa'] = summary_plot['LitterCHa'] + summary_plot['AbcHa']
                else:
                    print('WARNING: EvalPlotSummaryCs - no litter data for ID: {0}'.format(id))
                    summary_plot['LitterC'] = np.nan
                    summary_plot['LitterCHa'] = np.nan
                    summary_plot['AgcHa'] = summary_plot['AbcHa']

            self.summary_plots[id] = summary_plot

    def WriteAllCsFile(self, out_file_name=None):
        if len(self.plots) == 0:
            print('EvalAllRecordCs has not been called')
            return

        if out_file_name is None:
            out_file_name = str.format(str('{0}/{1} - All WoodyC.csv'), os.path.dirname(self.woody_file_name),
                                     os.path.splitext(os.path.basename(self.woody_file_name))[0])
        print('Writing plot data to: {0}'.format(out_file_name))
        with open(out_file_name,'w', newline='') as outfile:
            writer = DictWriter(outfile, list(list(self.plots.values())[0][0].keys()))
            writer.writeheader()
            for plot in list(self.plots.values()):
                writer.writerows(plot)

    def WriteSummaryFile(self, out_file_name=None):
        if len(self.summary_plots) == 0:
            print('EvalPlotSummaryCs has not been called')
            return

        if out_file_name is None:
            out_file_name = str.format(str('{0}/{1} - Summary WoodyC & LitterC.csv'), os.path.dirname(self.woody_file_name),
                                     os.path.splitext(os.path.basename(self.woody_file_name))[0])
        print('Writing plot summary to: {0}'.format(out_file_name))
        with open(out_file_name, 'w', newline='') as outfile:
            writer = DictWriter(outfile, list(self.summary_plots.values())[0].keys())
            writer.writeheader()
            writer.writerows(list(self.summary_plots.values()))
