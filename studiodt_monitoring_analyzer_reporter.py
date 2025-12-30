""" 
    StudioDT
    Monitoring analyzer and reporter
    Francesco Melchiori, 2025
"""


import os
import shelve
import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from pat_studiodt_monitoraggio_scuole_infanzia import pat_studiodt_misurazione_2024_25 as pat_sdt_msr_24_25


BINS_NORM_METRIC = 20
NUMBER_SCHOOL_CHART = 20


PATH_SCUOLE_INFANZIA = os.path.abspath('' + \
    'C:\\franzmelchiori\\projects\\studiodt' + \
    '\\pat_studiodt_monitoraggio_scuole_infanzia')
PATH_SCUOLE_INFANZIA_GRAFICI_2024_25 = os.path.abspath('' + \
    PATH_SCUOLE_INFANZIA + \
    '\\grafici_2024_25')
FILENAME_PREFIX_SCUOLE_INFANZIA_GRAFICI_2024_25 = \
    '\\pat_studiodt_monitoraggio_scuole_infanzia_2024_25_'


# INDEXES
#     'nome_scuola_file'
#     'nome_scuola'
#     'cod_b'

# GENERAL FEATURES
#     'm_f'
#     'genitore_straniero'

# RILEVAZIONE 1
#     GENERAL FEATURES
#         'storia_coniglietto_bambino_seguito'
#         'storia_coniglietto_eta'
#     GENERAL NOTES
#         'storia_coniglietto_note'
LABELS_CNG_FEATURES_LEFT = [
    'storia_coniglietto_lessico',
    'storia_coniglietto_sintassi',
    'storia_coniglietto_pragmatica',
    'storia_coniglietto_m_verb_ripetizione',
    'storia_coniglietto_m_verb_domande',
    'storia_coniglietto_mf_fus',
    'storia_coniglietto_mf_seg']
LABELS_CNG_FEATURES_RIGHT = [
    'storia_coniglietto_numero_conteggio',
    'storia_coniglietto_numero_cardin',
    'storia_coniglietto_numero_enum_0_20',
    'storia_coniglietto_copia_casa',
    'storia_coniglietto_copia_orologio',
    'storia_coniglietto_copia_tot',
    'storia_coniglietto_m_spaz']
LABELS_CNG_FEATURES = \
    LABELS_CNG_FEATURES_LEFT + \
    LABELS_CNG_FEATURES_RIGHT

# RILEVAZIONE 2
#     GENERAL FEATURES
#         'storia_riccio_bambino_seguito'
#         'storia_riccio_eta'
#     GENERAL NOTES
#         'storia_riccio_note'
LABELS_RCC_FEATURES_LEFT = [
    'storia_riccio_lessico',
    'storia_riccio_sintassi',
    'storia_riccio_pragmatica',
    'storia_riccio_m_verb_ripetizione',
    'storia_riccio_m_verb_domande',
    'storia_riccio_mf_fus',
    'storia_riccio_mf_seg']
LABELS_RCC_FEATURES_RIGHT = [
    'storia_riccio_numero_conteggio',
    'storia_riccio_numero_cardin',
    'storia_riccio_numero_enum_0_20',
    'storia_riccio_copia_casa',
    'storia_riccio_copia_orologio',
    'storia_riccio_copia_tot',
    'storia_riccio_m_spaz']
LABELS_RCC_FEATURES = \
    LABELS_RCC_FEATURES_LEFT + \
    LABELS_RCC_FEATURES_RIGHT

LABELS_FEATURES = \
    LABELS_CNG_FEATURES + \
    LABELS_RCC_FEATURES


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    TOGGLE_FIG_SAVEFIG = True
    TOGGLE_PLT_SHOW = False

    dataframe_scuole_infanzia_2024_25 = pat_sdt_msr_24_25.ScuoleInfanziaXLSX(
        pat_sdt_msr_24_25.PATH_SCUOLE_INFANZIA_MISURE_2024_25,
        pat_sdt_msr_24_25.STUDIODT_SCUOLE_INFANZIA_EXPERIMENT_DESIGN_2024_25,
        toggle_load_dataframe=True).dataframe_scuole_infanzia

    nomi_scuole = dataframe_scuole_infanzia_2024_25['nome_scuola_file'].unique()
    numeri_circoli = dataframe_scuole_infanzia_2024_25['numero_circolo'].unique()
    label_features_rilevazione_1 = dataframe_scuole_infanzia_2024_25.columns[8:22]
    label_features_rilevazione_2 = dataframe_scuole_infanzia_2024_25.columns[25:39]

    colormap_viridis = mpl.colormaps['viridis'].colors
    colormap_turbo = mpl.colormaps['turbo'].colors

	# 1. rappresentazione territoriale
	#   - chart | provincia | pie | numero bimbi nei circoli (13) | ci sono grandi e piccoli circoli?
    dimensioni_circoli = []
    for numero_circolo in numeri_circoli:
        filter_club = dataframe_scuole_infanzia_2024_25['numero_circolo'] == numero_circolo
        dataframe_scuola_infanzia = dataframe_scuole_infanzia_2024_25[filter_club]
        dimensioni_circoli.append([len(dataframe_scuola_infanzia), 'C{}'.format(numero_circolo)])
    dimensioni_circoli.sort()
    fig, ax = plt.subplots()
    ax.pie([circolo[0] for circolo in dimensioni_circoli],
           colors=colormap_viridis[::int(len(colormap_viridis)/len(dimensioni_circoli))],
           labels=[circolo[1] for circolo in dimensioni_circoli],
           radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))
    plt.title('PAT | Bambini per circolo')
    if TOGGLE_FIG_SAVEFIG:
        fig.savefig(
            os.path.abspath(PATH_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                            FILENAME_PREFIX_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                            'chart_01a_provincia_pie_percentuale_bimbi_nei_circoli.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.25)
    if TOGGLE_PLT_SHOW:
        plt.show()
    plt.close()
	#   - chart | provincia | pie | numero bimbi nelle scuole (111) | ci sono grandi e piccole scuole?
    dimensioni_nomi_scuole = []
    for nome_scuola in nomi_scuole:
        filter_school = dataframe_scuole_infanzia_2024_25['nome_scuola_file'] == nome_scuola
        dataframe_scuola_infanzia = dataframe_scuole_infanzia_2024_25[filter_school]
        dimensioni_nomi_scuole.append([len(dataframe_scuola_infanzia), nome_scuola.replace('_', ' ').title()])
    dimensioni_nomi_scuole.sort()
    fig, ax = plt.subplots()
    ax.pie([scuola[0] for scuola in dimensioni_nomi_scuole],
           colors=colormap_viridis[::int(len(colormap_viridis)/len(dimensioni_nomi_scuole))],
           labels=[scuola[1] for scuola in dimensioni_nomi_scuole],
           radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))
    plt.title('PAT | Bambini per scuola')
    if TOGGLE_FIG_SAVEFIG:
        fig.savefig(
            os.path.abspath(PATH_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                            FILENAME_PREFIX_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                            'chart_01b_provincia_pie_percentuale_bimbi_nelle_scuole.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.25)
    if TOGGLE_PLT_SHOW:
        plt.show()
    plt.close()
    
    # 2. rappresentazione genere e provenienza (M, F)
	#   - chart | provincia | pie | percentuale bimbi per genere e provenienza (straniero) | ci sono squilibri di genere e per etnia?
    def chart_pie_bimbi_genere_provenienza(dataframe_bimbi, title_prefix='', toggle_fig_savefig=False, filename_suffix='chart_02_pie_percentuale_bimbi_genere_provenienza'):
        filter_isna_mf = dataframe_bimbi['m_f'].isna()
        filter_isna_gs = dataframe_bimbi['genitore_straniero'].isna()
        filter_isna = filter_isna_mf | filter_isna_gs
        filter_ism = (filter_isna == False) & (dataframe_bimbi['m_f'] == 'm') & (dataframe_bimbi['genitore_straniero'] == False)
        filter_ism_gs = (filter_isna == False) & (dataframe_bimbi['m_f'] == 'm') & (dataframe_bimbi['genitore_straniero'])
        filter_isf = (filter_isna == False) & (dataframe_bimbi['m_f'] == 'f') & (dataframe_bimbi['genitore_straniero'] == False)
        filter_isf_gs = (filter_isna == False) & (dataframe_bimbi['m_f'] == 'f') & (dataframe_bimbi['genitore_straniero'])
        fig, ax = plt.subplots()
        ax.pie([sum(filter_ism_gs), sum(filter_ism), sum(filter_isna), sum(filter_isf), sum(filter_isf_gs)],
            colors = ['lightblue', 'lightblue', 'lightgrey', 'lightpink', 'lightpink'],
            hatch = ['..', '', '', '', '..'],
            labels=['M straniero', 'M', 'nd', 'F', 'F straniera'],
            radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))
        plt.title(title_prefix + 'Bambini per genere')
        if toggle_fig_savefig:
            fig.savefig(
                os.path.abspath(PATH_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                                FILENAME_PREFIX_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                                filename_suffix),
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    chart_pie_bimbi_genere_provenienza(dataframe_scuole_infanzia_2024_25,
                                       'PAT | ', TOGGLE_FIG_SAVEFIG, 'chart_02a_provincia_pie_percentuale_bimbi_genere_provenienza')
    if TOGGLE_PLT_SHOW:
        plt.show()
    plt.close()
	#   - chart | circoli | pie | percentuale bimbi nel circolo per genere e provenienza (straniero) | ci sono squilibri di genere e per etnia?
	#   - TODO | next | chart | circoli | hist | percentuale bimbi nei circoli per genere e provenienza (straniero) | ci sono squilibri di genere, per etnia e fra circoli?
    for numero_circolo in numeri_circoli:
        filter_club = dataframe_scuole_infanzia_2024_25['numero_circolo'] == numero_circolo
        dataframe_bimbi = dataframe_scuole_infanzia_2024_25[filter_club]
        chart_pie_bimbi_genere_provenienza(dataframe_bimbi,
                                           'C{} | '.format(numero_circolo), TOGGLE_FIG_SAVEFIG, 'chart_02b_circolo_{}_pie_percentuale_bimbi_genere_provenienza'.format(numero_circolo))
        if TOGGLE_PLT_SHOW:
            plt.show()
        plt.close()

	# 3. rappresentazione sostegno
	# 	- chart | provincia | pie | numero bimbi seguiti dalla rilevazione 1 alla 2 | c'e' un cambiamento nel numero di bimbi seguiti?
    def chart_pie_bimbi_seguiti(dataframe_bimbi, title_prefix='', toggle_fig_savefig=False, filename_suffix='chart_03_pie_percentuale_bimbi_seguiti'):
        filter_isna_cs = dataframe_bimbi['storia_coniglietto_bambino_seguito'].isna()
        filter_isna_rs = dataframe_bimbi['storia_riccio_bambino_seguito'].isna()
        filter_iscs = (filter_isna_cs == False) & (dataframe_bimbi['storia_coniglietto_bambino_seguito'])
        filter_isc = (filter_isna_cs == False) & (dataframe_bimbi['storia_coniglietto_bambino_seguito'] == False)
        filter_isrs = (filter_isna_rs == False) & (dataframe_bimbi['storia_riccio_bambino_seguito'])
        filter_isr = (filter_isna_rs == False) & (dataframe_bimbi['storia_riccio_bambino_seguito'] == False)
        fig, ax = plt.subplots()
        ax.pie([sum(filter_isrs), sum(filter_isr), sum(filter_isna_rs)],
            colors = ['orange', 'lightgreen', 'lightgrey'],
            labels=['riccio seguito', 'non seguito', 'nd'],
            radius=1, wedgeprops=dict(width=0.15, edgecolor='w'))
        ax.pie([sum(filter_iscs), sum(filter_isc), sum(filter_isna_cs)],
            colors = ['orange', 'lightgreen', 'lightgrey'],
            labels=['coniglietto seguito', '', ''],
            radius=1-0.15, wedgeprops=dict(width=0.15, edgecolor='w'))
        plt.title(title_prefix + 'Bambini seguiti')
        if toggle_fig_savefig:
            fig.savefig(
                os.path.abspath(PATH_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                                FILENAME_PREFIX_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                                filename_suffix),
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    chart_pie_bimbi_seguiti(dataframe_scuole_infanzia_2024_25,
                            'PAT | ', TOGGLE_FIG_SAVEFIG, 'chart_03a_provincia_pie_percentuale_bimbi_seguiti')
    if TOGGLE_PLT_SHOW:
        plt.show()
    plt.close()
    # 	- chart | circoli | pie | numero bimbi seguiti nel circolo dalla rilevazione 1 alla 2 | c'e' un cambiamento nel numero di bimbi seguiti nei circoli?
	# 	- TODO | next | chart | circoli | hist | numero bimbi seguiti nei circoli dalla rilevazione 1 alla 2 | c'e' un cambiamento nel numero di bimbi seguiti fra circoli?
    for numero_circolo in numeri_circoli:
        filter_club = dataframe_scuole_infanzia_2024_25['numero_circolo'] == numero_circolo
        dataframe_bimbi = dataframe_scuole_infanzia_2024_25[filter_club]
        chart_pie_bimbi_seguiti(dataframe_bimbi,
                                'C{} | '.format(numero_circolo), TOGGLE_FIG_SAVEFIG, 'chart_03b_circolo_{}_pie_percentuale_bimbi_seguiti'.format(numero_circolo))
        if TOGGLE_PLT_SHOW:
            plt.show()
        plt.close()

	# 4. rappresentazione eta'
    #   - chart | provincia | hist | eta' bimbi rilevazione 1 e 2 (mesi) | qual e' il cambiamento nella distribuzione dell'eta'?
    def chart_hist_bimbi_eta(dataframe_bimbi, vlines_height=200, title_prefix='', toggle_fig_savefig=False, filename_suffix='chart_04_hist_distribuzione_bimbi_eta'):
        filter_isna_ce = dataframe_bimbi['storia_coniglietto_eta'].isna()
        filter_isna_re = dataframe_bimbi['storia_riccio_eta'].isna()
        eta_c = dataframe_bimbi.loc[(filter_isna_ce == False), 'storia_coniglietto_eta']
        eta_r = dataframe_bimbi.loc[(filter_isna_re == False), 'storia_riccio_eta']
        fig, ax = plt.subplots()
        ax.hist(eta_c, bins=int(eta_c.max()-eta_c.min()), label='coniglietto', color='blue', alpha=0.5)
        ax.hist(eta_r, bins=int(eta_r.max()-eta_r.min()), label='riccio', color='green', alpha=0.5)
        ax.set_xlabel('mesi')
        ax.set_ylabel('numero bambini')
        ax.spines.left.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.legend()
        ax.vlines(eta_c.mean(), 0, vlines_height, color='blue', label=str(math.ceil(eta_c.mean())))
        ax.annotate(str(math.ceil(eta_c.mean())), (eta_c.mean(), vlines_height))
        ax.vlines(eta_r.mean(), 0, vlines_height, color='green', label=str(math.ceil(eta_c.mean())))
        ax.annotate(str(math.ceil(eta_r.mean())), (eta_r.mean(), vlines_height))
        plt.title(title_prefix + 'Età bambini')
        if toggle_fig_savefig:
            fig.savefig(
                os.path.abspath(PATH_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                                FILENAME_PREFIX_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                                filename_suffix),
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    chart_hist_bimbi_eta(dataframe_scuole_infanzia_2024_25,
                         200, 'PAT | ', TOGGLE_FIG_SAVEFIG, 'chart_04a_provincia_hist_distribuzione_bimbi_eta')
    if TOGGLE_PLT_SHOW:
        plt.show()
    plt.close()
    #   - chart | circoli | hist | eta' bimbi rilevazione 1 e 2 (mesi) nei circoli | qual e' il cambiamento nella distribuzione dell'eta'?
    for numero_circolo in numeri_circoli:
        filter_club = dataframe_scuole_infanzia_2024_25['numero_circolo'] == numero_circolo
        dataframe_bimbi = dataframe_scuole_infanzia_2024_25[filter_club]
        chart_hist_bimbi_eta(dataframe_bimbi,
                             25, 'C{} | '.format(numero_circolo), TOGGLE_FIG_SAVEFIG, 'chart_04b_circolo_{}_hist_distribuzione_bimbi_eta'.format(numero_circolo))
        if TOGGLE_PLT_SHOW:
            plt.show()
        plt.close()

	# 5. rappresentazione metriche
	# 	- chart | provincia | hist | distribuzioni dei bimbi per ogni metrica, rilevazione 1 e 2 | ci sono metriche che piu' marcatamente cambiano (in meglio o in peggio)?
    def chart_hist_distribuzioni_metriche(dataframe_bimbi, title_prefix='', toggle_fig_savefig=False, filename_suffix='chart_05_hist_distribuzioni_metriche'):
        ids_features = list(range(len(LABELS_CNG_FEATURES)))
        fig, axs = plt.subplots(len(ids_features), 1)
        for id_feat in ids_features:
            label_feature_cng = LABELS_CNG_FEATURES[id_feat]
            label_feature_rcc = LABELS_RCC_FEATURES[id_feat]
            label_feature = label_feature_cng.removeprefix('storia_coniglietto_').replace('_', ' ')
            filter_isna_cng = dataframe_bimbi[label_feature_cng].isna()
            filter_isna_rcc = dataframe_bimbi[label_feature_rcc].isna()
            feature_cng = dataframe_bimbi.loc[(filter_isna_cng == False), label_feature_cng]
            feature_rcc = dataframe_bimbi.loc[(filter_isna_rcc == False), label_feature_rcc]
            axs[id_feat].hist(feature_cng, bins=BINS_NORM_METRIC, label='coniglietto', color='blue', alpha=0.5)
            axs[id_feat].hist(feature_rcc, bins=BINS_NORM_METRIC, label='riccio', color='green', alpha=0.5)
            axs[id_feat].set_ylabel(label_feature, rotation='horizontal', ha='right')
            if id_feat == 0:
                axs[id_feat].set_title(title_prefix + 'StudioDT Protocollo 2024-25')
            if id_feat == ids_features[-1]:
                axs[id_feat].legend(loc='lower left')
                axs[id_feat].set_xlabel('media punteggio metrica')
            axs[id_feat].spines.left.set_visible(False)
            axs[id_feat].spines.right.set_visible(False)
            axs[id_feat].spines.top.set_visible(False)
            axs[id_feat].spines.bottom.set_visible(False)
        if toggle_fig_savefig:
            fig.savefig(
                os.path.abspath(PATH_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                                FILENAME_PREFIX_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                                filename_suffix),
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    chart_hist_distribuzioni_metriche(dataframe_scuole_infanzia_2024_25,
                                      'PAT | ', TOGGLE_FIG_SAVEFIG, 'chart_05a_provincia_hist_distribuzioni_metriche')
    if TOGGLE_PLT_SHOW:
        plt.show()
    plt.close()
	# 	- chart | circoli | hist | distribuzioni dei bimbi per ogni metrica, rilevazione 1 e 2
	# 	- TODO | next | chart | circoli | boxplot | distribuzioni metriche rilevazione 1 e 2 nei circoli
    for numero_circolo in numeri_circoli:
        filter_club = dataframe_scuole_infanzia_2024_25['numero_circolo'] == numero_circolo
        dataframe_bimbi = dataframe_scuole_infanzia_2024_25[filter_club]
        chart_hist_distribuzioni_metriche(dataframe_bimbi,
                                          'C{} | '.format(numero_circolo), TOGGLE_FIG_SAVEFIG, 'chart_05b_circolo_{}_hist_distribuzioni_metriche'.format(numero_circolo))
        if TOGGLE_PLT_SHOW:
            plt.show()
        plt.close()
	# 	- TODO | next | chart | bambini | radar | distribuzioni metriche rilevazione 1 e 2 | bimbi raggruppati per scuole | scuola per scuola, ci sono gruppi di bimbi che piu' marcatamente cambiano (in meglio o in peggio)?
    # best_scuola_infanzia_rcc = select_scuole_infanzia_rcc.index[0]
    # filter_isbs_rcc = dataframe_scuole_infanzia_2024_25['nome_scuola_file'] == best_scuola_infanzia_rcc
    # dataframe_best_scuola_infanzia_rcc = dataframe_scuole_infanzia_2024_25[filter_isbs_rcc]
    # data = [
    #     (dataframe_best_scuola_infanzia_rcc['cod_b'].iloc[0], [
    #         dataframe_best_scuola_infanzia_rcc[LABELS_RCC_FEATURES].iloc[0].values,
    #         dataframe_best_scuola_infanzia_rcc[LABELS_CNG_FEATURES].iloc[0].values]),
    #     (dataframe_best_scuola_infanzia_rcc['cod_b'].iloc[1], [
    #         dataframe_best_scuola_infanzia_rcc[LABELS_RCC_FEATURES].iloc[1].values,
    #         dataframe_best_scuola_infanzia_rcc[LABELS_CNG_FEATURES].iloc[1].values]),
    #     (dataframe_best_scuola_infanzia_rcc['cod_b'].iloc[2], [
    #         dataframe_best_scuola_infanzia_rcc[LABELS_RCC_FEATURES].iloc[2].values,
    #         dataframe_best_scuola_infanzia_rcc[LABELS_CNG_FEATURES].iloc[2].values]),
    #     (dataframe_best_scuola_infanzia_rcc['cod_b'].iloc[3], [
    #         dataframe_best_scuola_infanzia_rcc[LABELS_RCC_FEATURES].iloc[3].values,
    #         dataframe_best_scuola_infanzia_rcc[LABELS_CNG_FEATURES].iloc[3].values])]
    # N = 14
    # theta = radar_factory(N, frame='polygon')
    # spoke_labels = [label.removeprefix('storia_coniglietto_').replace('_', ' ') for label in LABELS_CNG_FEATURES]
    # fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
    #                         subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    # colors = ['r', 'y']
    ## Plot the four cases from the example data on separate Axes
    # print_spoke_labels = True
    # for ax, (title, case_data) in zip(axs.flat, data):
    #     ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    #     ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
    #                  horizontalalignment='center', verticalalignment='center')
    #     for d, color in zip(case_data, colors):
    #         ax.plot(theta, d, color=color)
    #         ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
    #     if print_spoke_labels:
    #         ax.set_varlabels(spoke_labels)
    #         print_spoke_labels = False
    #     else:
    #         ax.set_varlabels('')
    ## add legend relative to top-left plot
    # legend = axs[0, 0].legend(('riccio', 'coniglietto'), loc=(0.9, .95),
    #                        labelspacing=0.1, fontsize='small')
    # fig.text(0.5, 0.965, 'StudioDT Protocollo 2024-25 [0-100]',
    #          horizontalalignment='center', color='black', weight='bold', size='large')
    # fig.text(0.5, 0.935, best_scuola_infanzia_rcc.replace('_', ' ').title(),
    #          horizontalalignment='center', color='black', size='large')
    # worst_scuola_infanzia_rcc = select_scuole_infanzia_rcc.index[7]
    # filter_isbs_rcc = dataframe_scuole_infanzia_2024_25['nome_scuola_file'] == worst_scuola_infanzia_rcc
    # dataframe_worst_scuola_infanzia_rcc = dataframe_scuole_infanzia_2024_25[filter_isbs_rcc]
    # data = [
    #     (dataframe_worst_scuola_infanzia_rcc['cod_b'].iloc[0], [
    #         dataframe_worst_scuola_infanzia_rcc[LABELS_RCC_FEATURES].iloc[0].values,
    #         dataframe_worst_scuola_infanzia_rcc[LABELS_CNG_FEATURES].iloc[0].values]),
    #     (dataframe_worst_scuola_infanzia_rcc['cod_b'].iloc[1], [
    #         dataframe_worst_scuola_infanzia_rcc[LABELS_RCC_FEATURES].iloc[1].values,
    #         dataframe_worst_scuola_infanzia_rcc[LABELS_CNG_FEATURES].iloc[1].values]),
    #     (dataframe_worst_scuola_infanzia_rcc['cod_b'].iloc[2], [
    #         dataframe_worst_scuola_infanzia_rcc[LABELS_RCC_FEATURES].iloc[2].values,
    #         dataframe_worst_scuola_infanzia_rcc[LABELS_CNG_FEATURES].iloc[2].values]),
    #     (dataframe_worst_scuola_infanzia_rcc['cod_b'].iloc[3], [
    #         dataframe_worst_scuola_infanzia_rcc[LABELS_RCC_FEATURES].iloc[3].values,
    #         dataframe_worst_scuola_infanzia_rcc[LABELS_CNG_FEATURES].iloc[3].values])]
    # N = 14
    # theta = radar_factory(N, frame='polygon')
    # spoke_labels = [label.removeprefix('storia_coniglietto_').replace('_', ' ') for label in LABELS_CNG_FEATURES]
    # fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
    #                         subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    # colors = ['r', 'y']
    ## Plot the four cases from the example data on separate Axes
    # print_spoke_labels = True
    # for ax, (title, case_data) in zip(axs.flat, data):
    #     ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    #     ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
    #                  horizontalalignment='center', verticalalignment='center')
    #     for d, color in zip(case_data, colors):
    #         ax.plot(theta, d, color=color)
    #         ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
    #     if print_spoke_labels:
    #         ax.set_varlabels(spoke_labels)
    #         print_spoke_labels = False
    #     else:
    #         ax.set_varlabels('')
    ## add legend relative to top-left plot
    # legend = axs[0, 0].legend(('riccio', 'coniglietto'), loc=(0.9, .95),
    #                        labelspacing=0.1, fontsize='small')
    # fig.text(0.5, 0.965, 'StudioDT Protocollo 2024-25 [0-100]',
    #          horizontalalignment='center', color='black', weight='bold', size='large')
    # fig.text(0.5, 0.935, worst_scuola_infanzia_rcc.replace('_', ' ').title(),
    #          horizontalalignment='center', color='black', size='large')
    # if TOGGLE_PLT_SHOW:
    #     plt.show()
    # plt.close()

	# 6. rappresentazione performance
	# 	- deprecated | chart | scuole migliori e peggiori | hist | performance scuole migliori e peggiori rilevazione 1 e 2 | ci sono circoli che piu' marcatamente cambiano (in meglio o in peggio)?
    # performance_scuole_infanzia_rcc = []
    # for nome_scuola in nomi_scuole:
    #     filter_issc = dataframe_scuole_infanzia_2024_25['nome_scuola_file'] == nome_scuola
    #     performance_scuole_infanzia_rcc.append(dataframe_scuole_infanzia_2024_25.loc[filter_issc, LABELS_RCC_FEATURES].mean(axis=1).mean())
    # series_performance_scuole_infanzia_rcc = pd.Series(performance_scuole_infanzia_rcc, index=nomi_scuole).sort_values(ascending=False)
    # best_scuole_infanzia_rcc = series_performance_scuole_infanzia_rcc[:int(NUMBER_SCHOOL_CHART/2)]
    # worst_scuole_infanzia_rcc = series_performance_scuole_infanzia_rcc[-int(NUMBER_SCHOOL_CHART/2):]
    # select_scuole_infanzia_rcc = pd.concat([best_scuole_infanzia_rcc, worst_scuole_infanzia_rcc])
    # id_scuola = 0
    # fig, axs = plt.subplots(len(select_scuole_infanzia_rcc), 1)
    # for nome_scuola in select_scuole_infanzia_rcc.index:
    #     filter_issc = dataframe_scuole_infanzia_2024_25['nome_scuola_file'] == nome_scuola
    #     dataframe_scuola_infanzia_cng = dataframe_scuole_infanzia_2024_25.loc[filter_issc, LABELS_CNG_FEATURES].mean(axis=1)
    #     dataframe_scuola_infanzia_rcc = dataframe_scuole_infanzia_2024_25.loc[filter_issc, LABELS_RCC_FEATURES].mean(axis=1)
    #     filter_isna_cng = dataframe_scuola_infanzia_cng.isna()
    #     filter_isna_rcc = dataframe_scuola_infanzia_rcc.isna()
    #     axs[id_scuola].hist(dataframe_scuola_infanzia_cng.loc[filter_isna_cng == False], bins=BINS_NORM_METRIC, range=[0, pat_sdt_msr_24_25.MAX_NORM_METRIC], label='coniglietto [media metriche]', color='blue', alpha=0.5)
    #     axs[id_scuola].hist(dataframe_scuola_infanzia_rcc.loc[filter_isna_rcc == False], bins=BINS_NORM_METRIC, range=[0, pat_sdt_msr_24_25.MAX_NORM_METRIC], label='riccio [media metriche]', color='green', alpha=0.5)
    #     if pd.isna(dataframe_scuola_infanzia_cng.mean()) == False:
    #         axs[id_scuola].vlines(dataframe_scuola_infanzia_cng.mean(), 0, 10, color='blue')
    #         axs[id_scuola].annotate(str(math.ceil(dataframe_scuola_infanzia_cng.mean())), (dataframe_scuola_infanzia_cng.mean(), 10))
    #     if pd.isna(dataframe_scuola_infanzia_rcc.mean()) == False:
    #         axs[id_scuola].vlines(dataframe_scuola_infanzia_rcc.mean(), 0, 10, color='green')
    #         axs[id_scuola].annotate(str(math.ceil(dataframe_scuola_infanzia_rcc.mean())), (dataframe_scuola_infanzia_rcc.mean(), 10))
    #     if id_scuola == 0:
    #         axs[id_scuola].legend()
    #         axs[id_scuola].set_title('StudioDT Protocollo 2024-25 [0-100]')
    #     axs[id_scuola].set_ylabel(nome_scuola.title().replace('_', ' '), rotation='horizontal', ha='right')
    #     id_scuola += 1
    # if TOGGLE_PLT_SHOW:
    #     plt.show()
    # plt.close()
	# 	- chart | provincia | hist | performance circoli rilevazione 1 e 2 | ci sono circoli che piu' marcatamente cambiano (in meglio o in peggio)?
	# 	- TODO | chart | provincia | scatter | performance circoli (colore e diametro dal numero dei bimbi) rilevazione 1 (y) e 2 (x)
	# 	- TODO | next | chart | provincia | scatter | performance circoli (colore e diametro dal numero dei bimbi) stranieri (%) rilevazione 1 (y) e 2 (x)
	# 	- TODO | next | chart | provincia | scatter | performance circoli (colore e diametro dal numero dei bimbi) seguiti (%) rilevazione 1 (y) e 2 (x)
    performance_circoli_rcc = []
    for numero_circolo in numeri_circoli:
        filter_club = dataframe_scuole_infanzia_2024_25['numero_circolo'] == numero_circolo
        dataframe_bimbi = dataframe_scuole_infanzia_2024_25[filter_club]
        performance_circoli_rcc.append(dataframe_scuole_infanzia_2024_25.loc[filter_club, LABELS_RCC_FEATURES].mean(axis=1).mean())
    series_performance_circoli_rcc = pd.Series(performance_circoli_rcc, index=numeri_circoli).sort_values(ascending=False)
    fig, axs = plt.subplots(len(numeri_circoli), 1)
    list_id_circoli = list(range(len(series_performance_circoli_rcc)))
    for id_circolo in list_id_circoli:
        numero_circolo = series_performance_circoli_rcc.index[id_circolo]
        filter_club = dataframe_scuole_infanzia_2024_25['numero_circolo'] == numero_circolo
        dataframe_bimbi = dataframe_scuole_infanzia_2024_25[filter_club]
        dataframe_bimbi_cng = dataframe_bimbi.loc[filter_club, LABELS_CNG_FEATURES].mean(axis=1)
        dataframe_bimbi_rcc = dataframe_bimbi.loc[filter_club, LABELS_RCC_FEATURES].mean(axis=1)
        filter_isna_cng = dataframe_bimbi_cng.isna()
        filter_isna_rcc = dataframe_bimbi_rcc.isna()
        axs[id_circolo].hist(dataframe_bimbi_cng.loc[filter_isna_cng == False], bins=BINS_NORM_METRIC, range=[0, pat_sdt_msr_24_25.MAX_NORM_METRIC], label='coniglietto', color='blue', alpha=0.5)
        axs[id_circolo].hist(dataframe_bimbi_rcc.loc[filter_isna_rcc == False], bins=BINS_NORM_METRIC, range=[0, pat_sdt_msr_24_25.MAX_NORM_METRIC], label='riccio', color='green', alpha=0.5)
        if pd.isna(dataframe_bimbi_cng.mean()) == False:
            axs[id_circolo].vlines(dataframe_bimbi_cng.mean(), 0, 50, color='blue')
            axs[id_circolo].annotate(str(math.ceil(dataframe_bimbi_cng.mean())), (dataframe_bimbi_cng.mean(), 50))
        if pd.isna(dataframe_bimbi_rcc.mean()) == False:
            axs[id_circolo].vlines(dataframe_bimbi_rcc.mean(), 0, 50, color='green')
            axs[id_circolo].annotate(str(math.ceil(dataframe_bimbi_rcc.mean())), (dataframe_bimbi_rcc.mean(), 50))
        axs[id_circolo].set_ylabel('C{}'.format(numero_circolo), rotation='horizontal', ha='right')
        if id_circolo == 0:
            axs[id_circolo].set_title('PAT | StudioDT Protocollo 2024-25')
        if id_circolo == list_id_circoli[-1]:
            axs[id_circolo].legend(loc='lower left')
            axs[id_circolo].set_xlabel('media punteggi metriche')
        axs[id_circolo].spines.left.set_visible(False)
        axs[id_circolo].spines.right.set_visible(False)
        axs[id_circolo].spines.top.set_visible(False)
        axs[id_circolo].spines.bottom.set_visible(False)
    if TOGGLE_FIG_SAVEFIG:
        fig.savefig(
            os.path.abspath(PATH_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                            FILENAME_PREFIX_SCUOLE_INFANZIA_GRAFICI_2024_25 + \
                            'chart_06a_provincia_hist_performance'),
            dpi=300, bbox_inches='tight', pad_inches=0.25)
    if TOGGLE_PLT_SHOW:
        plt.show()
    plt.close()
