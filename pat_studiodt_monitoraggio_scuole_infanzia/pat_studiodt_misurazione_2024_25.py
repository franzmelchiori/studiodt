""" 
    PAT StudioDT
    Monitoraggio Scuole dell'Infanzia
    Misurazione 2024-25
    Francesco Melchiori, 2025
"""


import os
import shelve

import numpy as np
import pandas as pd


PATH_SCUOLE_INFANZIA = os.path.abspath('' + \
    'C:\\franzmelchiori\\projects\\studiodt' + \
    '\\pat_studiodt_monitoraggio_scuole_infanzia')
PATH_SCUOLE_INFANZIA_MISURE_2024_25 = os.path.abspath('' + \
    PATH_SCUOLE_INFANZIA + \
    '\\misure_2024_25')
SCHOOL_NAME_FILE_SPLIT_KEYWORD = '_rilevazioni'
FILENAME_ERROR_CSV = '\\pat_studiodt_monitoraggio_scuole_infanzia_misure_2024_25_error_{}.csv'

MAX_NORM_METRIC = 100

STUDIODT_SCUOLE_INFANZIA_EXPERIMENT_DESIGN_2024_25 = {
    'column_labels': [
        'nome_scuola',
        'cod_b',
        'm_f',
        'genitore_straniero',
        'storia_coniglietto_bambino_seguito',
        'storia_coniglietto_eta',
        'storia_coniglietto_note',
        'storia_coniglietto_lessico',
        'storia_coniglietto_sintassi',
        'storia_coniglietto_pragmatica',
        'storia_coniglietto_m_verb_ripetizione',
        'storia_coniglietto_m_verb_domande',
        'storia_coniglietto_mf_fus',
        'storia_coniglietto_mf_seg',
        'storia_coniglietto_numero_conteggio',
        'storia_coniglietto_numero_cardin',
        'storia_coniglietto_numero_enum_0_20',
        'storia_coniglietto_copia_casa',
        'storia_coniglietto_copia_orologio',
        'storia_coniglietto_copia_tot',
        'storia_coniglietto_m_spaz',
        'storia_riccio_bambino_seguito',
        'storia_riccio_eta',
        'storia_riccio_note',
        'storia_riccio_lessico',
        'storia_riccio_sintassi',
        'storia_riccio_pragmatica',
        'storia_riccio_m_verb_ripetizione',
        'storia_riccio_m_verb_domande',
        'storia_riccio_mf_fus',
        'storia_riccio_mf_seg',
        'storia_riccio_numero_conteggio',
        'storia_riccio_numero_cardin',
        'storia_riccio_numero_enum_0_20',
        'storia_riccio_copia_casa',
        'storia_riccio_copia_orologio',
        'storia_riccio_copia_tot',
        'storia_riccio_m_spaz'
    ],
    'column_indexes': list(range(21)) + list(range(25, 42)),
    # 'column_dtype': {
    #     'nome_scuola': str,
    #     'cod_b': str,
    #     'm_f': str,
    #     'genitore_straniero': str,
    #     'storia_coniglietto_bambino_seguito': str,
    #     'storia_coniglietto_eta': str,  # float
    #     'storia_coniglietto_note': str,
    #     'storia_coniglietto_lessico': float,
    #     'storia_coniglietto_sintassi': float,
    #     'storia_coniglietto_pragmatica': float,
    #     'storia_coniglietto_m_verb_ripetizione': float,
    #     'storia_coniglietto_m_verb_domande': float,
    #     'storia_coniglietto_mf_fus': float,
    #     'storia_coniglietto_mf_seg': float,
    #     'storia_coniglietto_numero_conteggio': float,
    #     'storia_coniglietto_numero_cardin': float,
    #     'storia_coniglietto_numero_enum_0_20': float,
    #     'storia_coniglietto_copia_casa': float,
    #     'storia_coniglietto_copia_orologio': float,
    #     'storia_coniglietto_copia_tot': float,
    #     'storia_coniglietto_m_spaz': float,
    #     'storia_riccio_bambino_seguito': str,
    #     'storia_riccio_eta': str,  # float
    #     'storia_riccio_note': str,
    #     'storia_riccio_lessico': float,
    #     'storia_riccio_sintassi': float,
    #     'storia_riccio_pragmatica': float,
    #     'storia_riccio_m_verb_ripetizione': float,
    #     'storia_riccio_m_verb_domande': float,
    #     'storia_riccio_mf_fus': float,
    #     'storia_riccio_mf_seg': float,
    #     'storia_riccio_numero_conteggio': float,
    #     'storia_riccio_numero_cardin': float,
    #     'storia_riccio_numero_enum_0_20': float,
    #     'storia_riccio_copia_casa': float,
    #     'storia_riccio_copia_orologio': float,
    #     'storia_riccio_copia_tot': float,
    #     'storia_riccio_m_spaz': float
    # },
    'row_skips': 3
}


class ScuolaInfanziaXLSX:

    def __init__(self, name_abspath_xlsx_file, dict_studiodt_experiment_design, toggle_check_read=False):
        self.school_name = name_abspath_xlsx_file[0]
        self.abspath_xlsx_file = os.path.abspath(name_abspath_xlsx_file[1])
        self.dict_studiodt_experiment_design = dict_studiodt_experiment_design
        self.toggle_check_read = toggle_check_read
        if self.toggle_check_read:
            self.check_xlsx_header()
        else:
            self.read_xlsx_file()
            # self.change_data_type()
    
    def check_xlsx_header(self):
        print('Read .xlsx of ' + self.school_name)
        self.dataframe_scuola_infanzia_header = pd.read_excel(
            io=self.abspath_xlsx_file,
            header=None,
            usecols=self.dict_studiodt_experiment_design['column_indexes'],
            skiprows=1,
            nrows=1)
        series_nome_scuola_file = pd.Series(
            [self.school_name for row in range(len(self.dataframe_scuola_infanzia_header))],
            name='nome_scuola_file')
        self.dataframe_scuola_infanzia_header = pd.concat(
            [series_nome_scuola_file, self.dataframe_scuola_infanzia_header],
            axis=1)
        return True

    def read_xlsx_file(self):
        print('Read .xlsx of ' + self.school_name)
        self.dataframe_scuola_infanzia_misura = pd.read_excel(
            io=self.abspath_xlsx_file,
            header=None,
            names=self.dict_studiodt_experiment_design['column_labels'],
            usecols=self.dict_studiodt_experiment_design['column_indexes'],
            # dtype=self.dict_studiodt_experiment_design['column_dtype'],
            skiprows=self.dict_studiodt_experiment_design['row_skips'])
        series_nome_scuola_file = pd.Series(
            [self.school_name for row in range(len(self.dataframe_scuola_infanzia_misura))],
            name='nome_scuola_file')
        self.dataframe_scuola_infanzia_misura = pd.concat(
            [series_nome_scuola_file, self.dataframe_scuola_infanzia_misura],
            axis=1)
        return True
        
    # def change_data_type(self):
    #     filter_mask = self.dataframe_scuola_infanzia_misura['genitore_straniero'] == '1'
    #     self.dataframe_scuola_infanzia_misura.loc[filter_mask, 'genitore_straniero'] = True
    #     filter_mask = self.dataframe_scuola_infanzia_misura['genitore_straniero'] == '0'
    #     self.dataframe_scuola_infanzia_misura.loc[filter_mask, 'genitore_straniero'] = False

    #     filter_mask = self.dataframe_scuola_infanzia_misura['storia_coniglietto_bambino_seguito'] == '1'
    #     self.dataframe_scuola_infanzia_misura.loc[filter_mask, 'storia_coniglietto_bambino_seguito'] = True
    #     filter_mask = self.dataframe_scuola_infanzia_misura['storia_coniglietto_bambino_seguito'] == '0'
    #     self.dataframe_scuola_infanzia_misura.loc[filter_mask, 'storia_coniglietto_bambino_seguito'] = False

    #     filter_mask = self.dataframe_scuola_infanzia_misura['storia_riccio_bambino_seguito'] == '1'
    #     self.dataframe_scuola_infanzia_misura.loc[filter_mask, 'storia_riccio_bambino_seguito'] = True
    #     filter_mask = self.dataframe_scuola_infanzia_misura['storia_riccio_bambino_seguito'] == '0'
    #     self.dataframe_scuola_infanzia_misura.loc[filter_mask, 'storia_riccio_bambino_seguito'] = False


class ScuoleInfanziaXLSX:

    def __init__(self, abspath_xlsx_files, dict_studiodt_experiment_design, toggle_load_dataframe=True, toggle_check_header=False, toggle_print_errors=False):
        self.abspath_xlsx_files = os.path.abspath(abspath_xlsx_files)
        self.dict_studiodt_experiment_design = dict_studiodt_experiment_design
        self.toggle_load_dataframe = toggle_load_dataframe
        self.toggle_check_header = toggle_check_header
        self.toggle_print_errors = toggle_print_errors
        if self.toggle_check_header:
            self.get_xlsx_paths()
            self.read_xlsx_files()
        else:
            if self.toggle_load_dataframe:
                self.load_dataframe()
            else:
                self.get_xlsx_paths()
                self.read_xlsx_files()
                self.clean_dataframe()
                self.save_dataframe()
            self.bound_dataframe()  # [!] overwrites potential errors
            self.norm_dataframe()

    def get_xlsx_paths(self):
        self.xlsx_paths = [
            (element.split(SCHOOL_NAME_FILE_SPLIT_KEYWORD)[0], os.path.join(self.abspath_xlsx_files, element)) \
            for element in os.listdir(self.abspath_xlsx_files) if os.path.isfile(os.path.join(self.abspath_xlsx_files, element))]
        return True

    def read_xlsx_files(self):
        if self.toggle_check_header:
            dataframe_scuole_infanzia_header = []
            for xlsx_path in self.xlsx_paths:
                scuola_infanzia_xlsx = ScuolaInfanziaXLSX(xlsx_path, self.dict_studiodt_experiment_design, toggle_check_read=True)
                dataframe_scuole_infanzia_header.append(scuola_infanzia_xlsx.dataframe_scuola_infanzia_header)
            self.dataframe_scuole_infanzia_header = pd.concat(
                dataframe_scuole_infanzia_header,
                axis='rows', join='outer')
        else:
            dataframes_scuole_infanzia = []
            for xlsx_path in self.xlsx_paths:
                scuola_infanzia_xlsx = ScuolaInfanziaXLSX(xlsx_path, self.dict_studiodt_experiment_design)
                dataframes_scuole_infanzia.append(scuola_infanzia_xlsx.dataframe_scuola_infanzia_misura)
            self.dataframe_scuole_infanzia = pd.concat(
                dataframes_scuole_infanzia,
                axis='rows', join='outer')
            # 1st DataFrame reset index
            self.dataframe_scuole_infanzia.reset_index(drop=True, inplace=True)
            # rimuovere misura (i.e. Series) in assenza (i.e. NaN) di un valore per 'cod_b' (i.e. DataFrame column)
            filter_mask = self.dataframe_scuole_infanzia['cod_b'].isna()
            index_to_drop = self.dataframe_scuole_infanzia[filter_mask].index
            self.dataframe_scuole_infanzia.drop(index=index_to_drop, inplace=True)
            # 2nd DataFrame reset index
            self.dataframe_scuole_infanzia.reset_index(drop=True, inplace=True)
        return True
    
    def clean_dataframe(self):
        # m_f | string | M, F o NaN | prendere m_f rilevazione 1
        self.dataframe_scuole_infanzia['m_f'] = self.dataframe_scuole_infanzia['m_f'].astype('string')
        self.dataframe_scuole_infanzia['m_f'] = self.dataframe_scuole_infanzia['m_f'].str.lower()
        filter_isna = self.dataframe_scuole_infanzia['m_f'].isna()
        filter_islegit = self.dataframe_scuole_infanzia['m_f'].str.contains(r'[m, f]', na=False)
        filter_iserror = (filter_isna == False) & (filter_islegit == False)
        if sum(filter_iserror) > 0:
            path_filename_error_csv = (PATH_SCUOLE_INFANZIA + FILENAME_ERROR_CSV).format('m_f')
            self.dataframe_scuole_infanzia.loc[filter_iserror].to_csv(path_filename_error_csv)
            self.dataframe_scuole_infanzia.loc[filter_iserror, 'm_f'] = pd.NA
        # genitore_straniero | boolean | True, False, NaN | prendere m_f rilevazione 1 (inferenza seconda lingua)
        self.dataframe_scuole_infanzia['genitore_straniero'] = self.dataframe_scuole_infanzia['genitore_straniero'].astype('string')
        self.dataframe_scuole_infanzia['genitore_straniero'] = self.dataframe_scuole_infanzia['genitore_straniero'].str.lower()
        filter_isna = self.dataframe_scuole_infanzia['genitore_straniero'].isna()
        filter_islegit = self.dataframe_scuole_infanzia['genitore_straniero'].str.contains(r'[0, 1]', na=False)
        filter_iserror = (filter_isna == False) & (filter_islegit == False)
        if sum(filter_iserror) > 0:
            path_filename_error_csv = (PATH_SCUOLE_INFANZIA + FILENAME_ERROR_CSV).format('genitore_straniero')
            self.dataframe_scuole_infanzia.loc[filter_iserror].to_csv(path_filename_error_csv)
            self.dataframe_scuole_infanzia.loc[filter_iserror, 'genitore_straniero'] = pd.NA
            filter_isna = self.dataframe_scuole_infanzia['genitore_straniero'].isna()
        filter_is1 = self.dataframe_scuole_infanzia['genitore_straniero'].str.contains(r'[1]', na=False)
        filter_is0 = (filter_isna == False) & (filter_is1 == False)
        self.dataframe_scuole_infanzia['genitore_straniero'] = self.dataframe_scuole_infanzia['genitore_straniero'].astype('object')
        self.dataframe_scuole_infanzia.loc[filter_is1, 'genitore_straniero'] = True
        self.dataframe_scuole_infanzia.loc[filter_is0, 'genitore_straniero'] = False
        self.dataframe_scuole_infanzia['genitore_straniero'] = self.dataframe_scuole_infanzia['genitore_straniero'].astype('boolean')
        # storia_coniglietto_bambino_seguito | boolean | True, False, NA
        self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'] = self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].astype('string')
        self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'] = self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].str.lower()
        self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].replace('o', '0', inplace=True)  # correction
        self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].replace('si', '1', inplace=True)  # correction
        filter_isna = self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].isna()
        filter_islegit = self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].str.contains(r'[0, 1]', na=False)
        filter_iserror = (filter_isna == False) & (filter_islegit == False)
        if sum(filter_iserror) > 0:
            path_filename_error_csv = (PATH_SCUOLE_INFANZIA + FILENAME_ERROR_CSV).format('storia_coniglietto_bambino_seguito')
            self.dataframe_scuole_infanzia.loc[filter_iserror].to_csv(path_filename_error_csv)
            self.dataframe_scuole_infanzia.loc[filter_iserror, 'storia_coniglietto_bambino_seguito'] = pd.NA
            filter_isna = self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].isna()
        filter_is1 = self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].str.contains(r'[1]', na=False)
        filter_is0 = (filter_isna == False) & (filter_is1 == False)
        self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'] = self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].astype('object')
        self.dataframe_scuole_infanzia.loc[filter_is1, 'storia_coniglietto_bambino_seguito'] = True
        self.dataframe_scuole_infanzia.loc[filter_is0, 'storia_coniglietto_bambino_seguito'] = False
        self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'] = self.dataframe_scuole_infanzia['storia_coniglietto_bambino_seguito'].astype('boolean')
        # storia_riccio_bambino_seguito | boolean | True, False, NA
        self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'] = self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].astype('string')
        self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'] = self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].str.lower()
        self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].replace('o', '0', inplace=True)  # correction
        self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].replace('si', '1', inplace=True)  # correction
        filter_isna = self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].isna()
        filter_islegit = self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].str.contains(r'[0, 1]', na=False)
        filter_iserror = (filter_isna == False) & (filter_islegit == False)
        if sum(filter_iserror) > 0:
            path_filename_error_csv = (PATH_SCUOLE_INFANZIA + FILENAME_ERROR_CSV).format('storia_riccio_bambino_seguito')
            self.dataframe_scuole_infanzia.loc[filter_iserror].to_csv(path_filename_error_csv)
            self.dataframe_scuole_infanzia.loc[filter_iserror, 'storia_riccio_bambino_seguito'] = pd.NA
            filter_isna = self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].isna()
        filter_is1 = self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].str.contains(r'[1]', na=False)
        filter_is0 = (filter_isna == False) & (filter_is1 == False)
        self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'] = self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].astype('object')
        self.dataframe_scuole_infanzia.loc[filter_is1, 'storia_riccio_bambino_seguito'] = True
        self.dataframe_scuole_infanzia.loc[filter_is0, 'storia_riccio_bambino_seguito'] = False
        self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'] = self.dataframe_scuole_infanzia['storia_riccio_bambino_seguito'].astype('boolean')
        # storia_coniglietto_eta | integer | conversione in mesi ('1.0' = 12 mesi, '0.1' = 1 mese, '0.10' = 10 mesi), NA
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].astype('string')
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.lower()
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.replace('o', '0')  # correction
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.replace('due', '2')  # correction
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.replace('dieci', '10')  # correction
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.replace(',', '.')  # correction
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.replace(':', '.')  # correction
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.replace(' ', '')  # correction
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.replace('.00', '')  # correction
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.replace('[a-z]', '', regex=True)  # correction
        filter_isna = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].isna()
        filter_isdot = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].str.contains(r'[.]', na=False)
        filter_isnotdot = (filter_isna == False) & (filter_isdot == False)
        self.dataframe_scuole_infanzia.loc[filter_isdot, 'storia_coniglietto_eta'] = \
            self.dataframe_scuole_infanzia.loc[filter_isdot, 'storia_coniglietto_eta'].map(lambda x: str((int(x.split('.')[0])*12)+int(x.split('.')[1])), na_action='ignore')
        self.dataframe_scuole_infanzia.loc[filter_isnotdot, 'storia_coniglietto_eta'] = \
            self.dataframe_scuole_infanzia.loc[filter_isnotdot, 'storia_coniglietto_eta'].map(lambda x: str(int(x)*12), na_action='ignore')
        self.dataframe_scuole_infanzia['storia_coniglietto_eta'] = self.dataframe_scuole_infanzia['storia_coniglietto_eta'].astype('float')
        filter_age3 = self.dataframe_scuole_infanzia['storia_coniglietto_eta'] < (4*12)
        self.dataframe_scuole_infanzia.loc[filter_age3, 'storia_coniglietto_eta'] = pd.NA
        # storia_riccio_eta | integer | conversione in mesi ('1.0' = 12 mesi, '0.1' = 1 mese, '0.10' = 10 mesi), NA
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].astype('string')
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.lower()
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace('o', '0')  # correction
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace('due', '2')  # correction
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace('dieci', '10')  # correction
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace('50000000000', '5')  # correction
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace(',', '.')  # correction
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace(':', '.')  # correction
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace(' ', '')  # correction
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace('.00', '')  # correction
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.replace('[a-z]', '', regex=True)  # correction
        filter_isdate = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.contains(r'[-]', na=False)
        self.dataframe_scuole_infanzia.loc[filter_isdate, 'storia_riccio_eta'] = pd.NA
        filter_isna = self.dataframe_scuole_infanzia['storia_riccio_eta'].isna()
        filter_isdot = self.dataframe_scuole_infanzia['storia_riccio_eta'].str.contains(r'[.]', na=False)
        filter_isnotdot = (filter_isna == False) & (filter_isdot == False)
        self.dataframe_scuole_infanzia.loc[filter_isdot, 'storia_riccio_eta'] = \
            self.dataframe_scuole_infanzia.loc[filter_isdot, 'storia_riccio_eta'].map(lambda x: str((int(x.split('.')[0])*12)+int(x.split('.')[1])), na_action='ignore')
        self.dataframe_scuole_infanzia.loc[filter_isnotdot, 'storia_riccio_eta'] = \
            self.dataframe_scuole_infanzia.loc[filter_isnotdot, 'storia_riccio_eta'].map(lambda x: str(int(x)*12), na_action='ignore')
        self.dataframe_scuole_infanzia['storia_riccio_eta'] = self.dataframe_scuole_infanzia['storia_riccio_eta'].astype('float')
        filter_age3 = self.dataframe_scuole_infanzia['storia_riccio_eta'] < (4*12)
        self.dataframe_scuole_infanzia.loc[filter_age3, 'storia_riccio_eta'] = pd.NA
        # 'storia_coniglietto_lessico' | integer | *, NA | 0-inf | piu' e' meglio | emisfero sinistro
        self.dataframe_scuole_infanzia['storia_coniglietto_lessico'] = self.dataframe_scuole_infanzia['storia_coniglietto_lessico'].astype('float')
        # 'storia_coniglietto_sintassi' | integer | *, NA | 0-4 | piu' e' meglio | emisfero sinistro
        self.dataframe_scuole_infanzia['storia_coniglietto_sintassi'] = self.dataframe_scuole_infanzia['storia_coniglietto_sintassi'].astype('float')
        # 'storia_coniglietto_pragmatica' | float | *, NA | 0-9 | piu' e' meglio | emisfero sinistro
        self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'] = self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'].astype('string')
        self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'] = self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'].str.lower()
        self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'] = self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'].str.replace('1/2', '.5')
        self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'] = self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'].str.replace(',', '.')
        self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'] = self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'].str.replace('..', '.')
        self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'] = self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'].str.replace(' ', '')
        self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'] = self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'].str.replace('[a-z]', '', regex=True)
        self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'] = self.dataframe_scuole_infanzia['storia_coniglietto_pragmatica'].astype('float')
        # 'storia_coniglietto_m_verb_ripetizione' | integer | *, NA | 0-10 | piu' e' meglio | memoria verbale a breve termine = parole alta frequenza + parole bassa frequenza | emisfero sinistro
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'].astype('string')
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'].str.lower()
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'].str.replace('1/2', '.5')
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'].str.replace(',', '.')
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'].str.replace('..', '.')
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'].str.replace(' ', '')
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'].str.replace('[a-z]', '', regex=True)
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_ripetizione'].astype('float')
        # 'storia_coniglietto_m_verb_domande' | integer | *, NA | 0-6 | piu' e' meglio | memoria verbale di lavoro | emisfero sinistro
        self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_verb_domande'].astype('float')
        # 'storia_coniglietto_mf_fus' | integer | *, NA | 0-8 | piu' e' meglio | emisfero sinistro
        self.dataframe_scuole_infanzia['storia_coniglietto_mf_fus'] = self.dataframe_scuole_infanzia['storia_coniglietto_mf_fus'].astype('float')
        # 'storia_coniglietto_mf_seg' | integer | *, NA | 0-8 | piu' e' meglio | emisfero sinistro
        self.dataframe_scuole_infanzia['storia_coniglietto_mf_seg'] = self.dataframe_scuole_infanzia['storia_coniglietto_mf_seg'].astype('float')
        # 'storia_coniglietto_numero_conteggio' | integer | 0-12 | piu' e' meglio (attenzione errori, es. max 12, 13 no sense) | *, NA | emisfero destro
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_conteggio'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_conteggio'].astype('float')
        # 'storia_coniglietto_numero_cardin' | bool | True-False | True e' meglio (attenzione influenza errori sul conteggio) | *, NA | emisfero destro
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].astype('string')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].str.lower()
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].str.replace('si', '1')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].str.replace('no', '0')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].str.replace(',', '.')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].str.replace('..', '.')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].str.replace(' ', '')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].str.replace('[a-z]', '', regex=True)
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_cardin'].astype('float')
        # 'storia_coniglietto_numero_enum_0_20' | integer | 0-20 | piu' e' meglio | *, NA | emisfero destro
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].astype('string')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].str.lower()
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].str.replace('-20', '')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].str.replace(',', '.')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].str.replace('..', '.')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].str.replace(' ', '')
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].str.replace('[a-z]', '', regex=True)
        filter_isdate = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].str.contains(r'[-]', na=False)
        filter_isvoid = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] == ''
        self.dataframe_scuole_infanzia.loc[filter_isdate | filter_isvoid, 'storia_coniglietto_numero_enum_0_20'] = pd.NA
        self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_coniglietto_numero_enum_0_20'].astype('float')
        # 'storia_coniglietto_copia_casa' | integer | 0-16 | piu' e' meglio | *, NA | emisfero destro
        self.dataframe_scuole_infanzia['storia_coniglietto_copia_casa'] = self.dataframe_scuole_infanzia['storia_coniglietto_copia_casa'].astype('float')
        # 'storia_coniglietto_copia_orologio' | integer | 0-16 | piu' e' meglio | *, NA | emisfero destro
        self.dataframe_scuole_infanzia['storia_coniglietto_copia_orologio'] = self.dataframe_scuole_infanzia['storia_coniglietto_copia_orologio'].astype('float')
        # 'storia_coniglietto_copia_tot' | integer | 0-32 (copia_casa + copia_orologio) | piu' e' meglio | *, NA | emisfero destro
        self.dataframe_scuole_infanzia['storia_coniglietto_copia_tot'] = self.dataframe_scuole_infanzia['storia_coniglietto_copia_tot'].astype('float')
        # 'storia_coniglietto_m_spaz' | integer | 0-46 | piu' e' meglio | *, NA | emisfero destro
        self.dataframe_scuole_infanzia['storia_coniglietto_m_spaz'] = self.dataframe_scuole_infanzia['storia_coniglietto_m_spaz'].astype('float')
        # 'storia_riccio_lessico' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_lessico'] = self.dataframe_scuole_infanzia['storia_riccio_lessico'].astype('string')
        self.dataframe_scuole_infanzia['storia_riccio_lessico'] = self.dataframe_scuole_infanzia['storia_riccio_lessico'].str.lower()
        self.dataframe_scuole_infanzia['storia_riccio_lessico'] = self.dataframe_scuole_infanzia['storia_riccio_lessico'].str.replace(',', '.')
        self.dataframe_scuole_infanzia['storia_riccio_lessico'] = self.dataframe_scuole_infanzia['storia_riccio_lessico'].str.replace('..', '.')
        self.dataframe_scuole_infanzia['storia_riccio_lessico'] = self.dataframe_scuole_infanzia['storia_riccio_lessico'].str.replace(' ', '')
        self.dataframe_scuole_infanzia['storia_riccio_lessico'] = self.dataframe_scuole_infanzia['storia_riccio_lessico'].str.replace('[a-z]', '', regex=True)
        filter_isdate = self.dataframe_scuole_infanzia['storia_riccio_lessico'].str.contains(r'[-]', na=False)
        filter_isvoid = self.dataframe_scuole_infanzia['storia_riccio_lessico'] == ''
        self.dataframe_scuole_infanzia.loc[filter_isdate | filter_isvoid, 'storia_riccio_lessico'] = pd.NA
        self.dataframe_scuole_infanzia['storia_riccio_lessico'] = self.dataframe_scuole_infanzia['storia_riccio_lessico'].astype('float')
        # 'storia_riccio_sintassi' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_sintassi'] = self.dataframe_scuole_infanzia['storia_riccio_sintassi'].astype('float')
        # 'storia_riccio_pragmatica' | integer | 0-15 | piu' e' meglio | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_pragmatica'] = self.dataframe_scuole_infanzia['storia_riccio_pragmatica'].astype('float')
        # 'storia_riccio_m_verb_ripetizione' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_ripetizione'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_ripetizione'].astype('float')
        # 'storia_riccio_m_verb_domande' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].astype('string')
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].str.lower()
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].str.replace(',', '.')
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].str.replace('..', '.')
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].str.replace('*', '')
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].str.replace(' ', '')
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].str.replace('[a-z]', '', regex=True)
        filter_isdate = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].str.contains(r'[-]', na=False)
        filter_isvoid = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] == ''
        self.dataframe_scuole_infanzia.loc[filter_isdate | filter_isvoid, 'storia_riccio_m_verb_domande'] = pd.NA
        self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'] = self.dataframe_scuole_infanzia['storia_riccio_m_verb_domande'].astype('float')
        # 'storia_riccio_mf_fus' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_mf_fus'] = self.dataframe_scuole_infanzia['storia_riccio_mf_fus'].astype('float')
        # 'storia_riccio_mf_seg' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_mf_seg'] = self.dataframe_scuole_infanzia['storia_riccio_mf_seg'].astype('float')
        # 'storia_riccio_numero_conteggio' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_numero_conteggio'] = self.dataframe_scuole_infanzia['storia_riccio_numero_conteggio'].astype('float')
        # 'storia_riccio_numero_cardin' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].astype('string')
        self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].str.lower()
        self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].str.replace(',', '.')
        self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].str.replace('..', '.')
        self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].str.replace('*', '')
        self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].str.replace(' ', '')
        self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].str.replace('[a-z]', '', regex=True)
        filter_isdate = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].str.contains(r'[-]', na=False)
        filter_isvoid = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] == ''
        self.dataframe_scuole_infanzia.loc[filter_isdate | filter_isvoid, 'storia_riccio_numero_cardin'] = pd.NA
        self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'] = self.dataframe_scuole_infanzia['storia_riccio_numero_cardin'].astype('float')
        # 'storia_riccio_numero_enum_0_20' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_numero_enum_0_20'] = self.dataframe_scuole_infanzia['storia_riccio_numero_enum_0_20'].astype('float')
        # 'storia_riccio_copia_casa' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_copia_casa'] = self.dataframe_scuole_infanzia['storia_riccio_copia_casa'].astype('float')
        # 'storia_riccio_copia_orologio' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_copia_orologio'] = self.dataframe_scuole_infanzia['storia_riccio_copia_orologio'].astype('float')
        # 'storia_riccio_copia_tot' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_copia_tot'] = self.dataframe_scuole_infanzia['storia_riccio_copia_tot'].astype('float')
        # 'storia_riccio_m_spaz' | integer | *, NA
        self.dataframe_scuole_infanzia['storia_riccio_m_spaz'] = self.dataframe_scuole_infanzia['storia_riccio_m_spaz'].astype('float')
        return True
    
    def save_dataframe(self):
        scuole_infanzia_dataframe_shelve = shelve.open(self.abspath_xlsx_files + '\\dtf\\scuole_infanzia_dataframe')
        scuole_infanzia_dataframe_shelve['scuole_infanzia_dataframe'] = self.dataframe_scuole_infanzia
        scuole_infanzia_dataframe_shelve.close()
        return True

    def load_dataframe(self):
        scuole_infanzia_dataframe_shelve = shelve.open(self.abspath_xlsx_files + '\\dtf\\scuole_infanzia_dataframe')
        self.dataframe_scuole_infanzia = scuole_infanzia_dataframe_shelve['scuole_infanzia_dataframe']
        scuole_infanzia_dataframe_shelve.close()
        return True

    def bound_dataframe(self):
        def check_bounds(name_metric, upperbound, toggle_fit_bounds=False):
            filter_isna = self.dataframe_scuole_infanzia[name_metric].isna()
            filter_isng = self.dataframe_scuole_infanzia.loc[filter_isna == False, name_metric] < 0
            filter_ishi = self.dataframe_scuole_infanzia.loc[filter_isna == False, name_metric] > upperbound
            filter_isin = self.dataframe_scuole_infanzia.loc[(filter_isna == False) & (filter_isng == False), name_metric] <= upperbound
            if self.toggle_print_errors:
                print('{0} isna: {1}'.format(name_metric, sum(filter_isna)))
                print('{0} isng: {1}'.format(name_metric, sum(filter_isng)))  # attenzione
                print('{0} ishi: {1}'.format(name_metric, sum(filter_ishi)))  # attenzione
                print('{0} isin: {1}'.format(name_metric, sum(filter_isin)))
                print('{0} istt: {1}'.format(name_metric, sum(filter_isna)+sum(filter_isng)+sum(filter_ishi)+sum(filter_isin)))
                print('')
            if sum(filter_isng) > 0:
                if self.toggle_print_errors:
                    print('{0} isng:'.format(name_metric))
                    print(self.dataframe_scuole_infanzia.loc[(filter_isna == False) & filter_isng, name_metric])
                    print('')
                if toggle_fit_bounds:
                    self.dataframe_scuole_infanzia.loc[(filter_isna == False) & filter_isng, name_metric] = 0
            if sum(filter_ishi) > 0:
                if self.toggle_print_errors:
                    print('{0} ishi:'.format(name_metric))
                    print(self.dataframe_scuole_infanzia.loc[(filter_isna == False) & filter_ishi, name_metric])
                    print('')
                if toggle_fit_bounds:
                    self.dataframe_scuole_infanzia.loc[(filter_isna == False) & filter_ishi, name_metric] = upperbound
            return True
        
        # storia_coniglietto_lessico | integer | *, NA | 0-22-inf | piu' e' meglio | emisfero sinistro
        check_bounds('storia_coniglietto_lessico', 22, toggle_fit_bounds=True)
        # storia_coniglietto_sintassi | integer | *, NA | 0-4 | piu' e' meglio | emisfero sinistro
        check_bounds('storia_coniglietto_sintassi', 4, toggle_fit_bounds=True)
        # storia_coniglietto_pragmatica | float | *, NA | 0-9 | piu' e' meglio | emisfero sinistro
        check_bounds('storia_coniglietto_pragmatica', 9, toggle_fit_bounds=True)
        # storia_coniglietto_m_verb_ripetizione | integer | *, NA | 0-10 | piu' e' meglio | memoria verbale a breve termine = parole alta frequenza + parole bassa frequenza | emisfero sinistro
        check_bounds('storia_coniglietto_m_verb_ripetizione', 10, toggle_fit_bounds=True)
        # storia_coniglietto_m_verb_domande | integer | *, NA | 0-6 | piu' e' meglio | memoria verbale di lavoro | emisfero sinistro
        check_bounds('storia_coniglietto_m_verb_domande', 6, toggle_fit_bounds=True)
        # storia_coniglietto_mf_fus | integer | *, NA | 0-8 | piu' e' meglio | emisfero sinistro
        check_bounds('storia_coniglietto_mf_fus', 8, toggle_fit_bounds=True)
        # storia_coniglietto_mf_seg | integer | *, NA | 0-8 | piu' e' meglio | emisfero sinistro
        check_bounds('storia_coniglietto_mf_seg', 8, toggle_fit_bounds=True)
        # storia_coniglietto_numero_conteggio | integer | 0-12 | piu' e' meglio (attenzione errori, es. max 12, 13 no sense) | *, NA | emisfero destro
        check_bounds('storia_coniglietto_numero_conteggio', 12, toggle_fit_bounds=True)
        # storia_coniglietto_numero_cardin | bool | True-False | True e' meglio (attenzione influenza errori sul conteggio) | *, NA | emisfero destro
        check_bounds('storia_coniglietto_numero_cardin', 1, toggle_fit_bounds=True)
        # storia_coniglietto_numero_enum_0_20 | integer | 0-20 | piu' e' meglio | *, NA | emisfero destro
        check_bounds('storia_coniglietto_numero_enum_0_20', 20, toggle_fit_bounds=True)
        # storia_coniglietto_copia_casa | integer | 0-16 | piu' e' meglio | *, NA | emisfero destro
        check_bounds('storia_coniglietto_copia_casa', 16, toggle_fit_bounds=True)
        # storia_coniglietto_copia_orologio | integer | 0-16 | piu' e' meglio | *, NA | emisfero destro
        check_bounds('storia_coniglietto_copia_orologio', 16, toggle_fit_bounds=True)
        # storia_coniglietto_copia_tot | integer | 0-32 (copia_casa + copia_orologio) | piu' e' meglio | *, NA | emisfero destro
        check_bounds('storia_coniglietto_copia_tot', 32, toggle_fit_bounds=True)
        # storia_coniglietto_m_spaz | integer | 0-46 | piu' e' meglio | *, NA | emisfero destro
        check_bounds('storia_coniglietto_m_spaz', 46, toggle_fit_bounds=True)
        
        # storia_riccio_lessico | integer | *, NA
        check_bounds('storia_riccio_lessico', 22, toggle_fit_bounds=True)
        # storia_riccio_sintassi | integer | *, NA
        check_bounds('storia_riccio_sintassi', 4, toggle_fit_bounds=True)
        # storia_riccio_pragmatica | integer | 0-15 | piu' e' meglio | *, NA
        check_bounds('storia_riccio_pragmatica', 15, toggle_fit_bounds=True)
        # storia_riccio_m_verb_ripetizione | integer | *, NA
        check_bounds('storia_riccio_m_verb_ripetizione', 10, toggle_fit_bounds=True)
        # storia_riccio_m_verb_domande | integer | *, NA
        check_bounds('storia_riccio_m_verb_domande', 6, toggle_fit_bounds=True)
        # storia_riccio_mf_fus | integer | *, NA
        check_bounds('storia_riccio_mf_fus', 8, toggle_fit_bounds=True)
        # storia_riccio_mf_seg | integer | *, NA
        check_bounds('storia_riccio_mf_seg', 8, toggle_fit_bounds=True)
        # storia_riccio_numero_conteggio | integer | *, NA
        check_bounds('storia_riccio_numero_conteggio', 12, toggle_fit_bounds=True)
        # storia_riccio_numero_cardin | integer | *, NA
        check_bounds('storia_riccio_numero_cardin', 1, toggle_fit_bounds=True)
        # storia_riccio_numero_enum_0_20 | integer | *, NA
        check_bounds('storia_riccio_numero_enum_0_20', 20, toggle_fit_bounds=True)
        # storia_riccio_copia_casa | integer | *, NA
        check_bounds('storia_riccio_copia_casa', 16, toggle_fit_bounds=True)
        # storia_riccio_copia_orologio | integer | *, NA
        check_bounds('storia_riccio_copia_orologio', 16, toggle_fit_bounds=True)
        # storia_riccio_copia_tot | integer | *, NA
        check_bounds('storia_riccio_copia_tot', 32, toggle_fit_bounds=True)
        # storia_riccio_m_spaz | integer | *, NA
        check_bounds('storia_riccio_m_spaz', 46, toggle_fit_bounds=True)
        return True

    def norm_dataframe(self):
        def norm_metric(name_metric, max_metric):
            self.dataframe_scuole_infanzia[name_metric] = (self.dataframe_scuole_infanzia[name_metric]/max_metric)*MAX_NORM_METRIC
            return True
        
        metric_to_norm = [
            ('storia_coniglietto_lessico', 22),
            ('storia_coniglietto_sintassi', 4),
            ('storia_coniglietto_pragmatica', 9),
            ('storia_coniglietto_m_verb_ripetizione', 10),
            ('storia_coniglietto_m_verb_domande', 6),
            ('storia_coniglietto_mf_fus', 8),
            ('storia_coniglietto_mf_seg', 8),
            ('storia_coniglietto_numero_conteggio', 12),
            ('storia_coniglietto_numero_cardin', 1),
            ('storia_coniglietto_numero_enum_0_20', 20),
            ('storia_coniglietto_copia_casa', 16),
            ('storia_coniglietto_copia_orologio', 16),
            ('storia_coniglietto_copia_tot', 32),
            ('storia_coniglietto_m_spaz', 46),
            ('storia_riccio_lessico', 22),
            ('storia_riccio_sintassi', 4),
            ('storia_riccio_pragmatica', 15),
            ('storia_riccio_m_verb_ripetizione', 10),
            ('storia_riccio_m_verb_domande', 6),
            ('storia_riccio_mf_fus', 8),
            ('storia_riccio_mf_seg', 8),
            ('storia_riccio_numero_conteggio', 12),
            ('storia_riccio_numero_cardin', 1),
            ('storia_riccio_numero_enum_0_20', 20),
            ('storia_riccio_copia_casa', 16),
            ('storia_riccio_copia_orologio', 16),
            ('storia_riccio_copia_tot', 32),
            ('storia_riccio_m_spaz', 46)]
        for metric in metric_to_norm:
            norm_metric(metric[0], metric[1])
        return True


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # scuola_infanzia_xlsx = ScuolaInfanziaXLSX(
    #     ('RONCAFORT', PATH_SCUOLE_INFANZIA_MISURE_2024_25 + '\\RONCAFORT_rilevazioni_2024_2025.xlsx'),
    #     STUDIODT_SCUOLE_INFANZIA_EXPERIMENT_DESIGN_2024_25,
    #     toggle_check_read=False)
    # scuola_infanzia_xlsx.dataframe_scuola_infanzia_misura
    # scuola_infanzia_xlsx.dataframe_scuola_infanzia_header

    scuole_infanzia_xlsx = ScuoleInfanziaXLSX(
        PATH_SCUOLE_INFANZIA_MISURE_2024_25,
        STUDIODT_SCUOLE_INFANZIA_EXPERIMENT_DESIGN_2024_25,
        toggle_load_dataframe=True,
        toggle_check_header=False,
        toggle_print_errors=False)
    # scuole_infanzia_xlsx.dataframe_scuole_infanzia
    # scuole_infanzia_xlsx.dataframe_scuole_infanzia_header

    pass
