from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from pycor.utils.preprocess import clean_ontology, clean_frame, get_fig_value, get_main_sense

#from pycor.utils.vectors import vectorize

anno = pd.read_csv('data/hum_anno/15_12_2021.txt',
                   sep='\t',
                   encoding='utf-8',
                   na_values=['n', ' '])

citater = pd.read_csv('data/citat/citater_ren.tsv',
                      sep='\t',
                      encoding='utf-8',
                      usecols=['ddo_dannetsemid', 'citat'])

# anno = pd.read_csv('data/hum_anno/mellemfrekvente.txt',
#                    sep='\t',
#                    encoding='utf-8',
#                    na_values=['n', ' '])
#
# citater = pd.read_csv('H:/CST_COR/data/DDO_citater/citater_mellemfrekvente.tsv',
#                       sep='\t',
#                       encoding='utf-8')
citater = citater.groupby('ddo_dannetsemid').aggregate(' '.join)

# columns = ['score', 'ddo_entryid', 'ddo_lemma', 'ddo_homnr', 'ddo_ordklasse',
#        'ddo_dannetsemid', 'ddo_definition', 'ddo_genprox', 'ddo_bemaerk',
#        'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame', 'ddo_valens',
#        'ddo_kollokation', 'COR-bet.inventar', 'Hvorfor?', 'Ny ordklasse',
#        'ddo_betyd_nr', 'ddo_bet_tags', 'ddo_senselevel', 'ddo_plac',
#        'ddo_art_tags', 'ddo_bet', 'ddo_mwe_bet',
#        'Antal citater til bet/ddo_bet_doks', 'ddo_art_doks', 'ddo_udvalg',
#        'dn_id', 'dn_lemma', 'ddo_sublemma', 'ddb_nøgleord']

anno = anno.dropna(subset=['ddo_lemma', 'COR-bet.inventar', 'ddo_dannetsemid', 'ddo_betyd_nr'])
anno = anno[anno['COR-bet.inventar'] != '0']

anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')
anno = anno.merge(citater, how='outer', on='ddo_dannetsemid')
anno = anno.dropna(subset=['ddo_lemma', 'COR-bet.inventar'])


anno = anno.loc[:, ['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
                    'COR-bet.inventar', 'dn_id', 'ddo_betyd_nr', 'citat', 'ddo_bemaerk',
                    'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame', 'ddo_konbet', 'ddo_encykl']]

anno.columns = ['lemma', 'ordklasse', 'homnr', 'definition', 'genprox', 'kollokation', 'cor', 'dn_id', 'ddo_nr',
                'citat', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame', 'konbet', 'encykl']

# DanNet = load_obj('DanNet')
# DanNet = expand_synsets(expand_synsets(DanNet), s=False)
DataInstance = namedtuple('DataInstance', ['lemma', 'ordklasse', 'homnr', 'ddo_bet', 'main_sense', 'cor',
                          'onto', 'figurative'])

def create_dataset(annotations, infotypes=['def']):
    dataset = []

    for name, group in annotations.groupby(['lemma', 'ordklasse', 'homnr']):
        groupset = []
        #if len(group.index) > 5:
            #continue
        for row in group.itertuples():
            figurative = row.bemaerk if type(row.bemaerk) != float else ''
            instance = DataInstance(lemma=row.lemma,
                                    ordklasse=row.ordklasse,
                                    homnr=row.homnr,
                                    ddo_bet=row.ddo_nr,
                                    main_sense = get_main_sense(row.ddo_nr),
                                    cor=row.cor,
                                    onto=clean_ontology(row.onto1).union(clean_ontology(row.onto2)),
                                    figurative=1 if 'ofø' in figurative else 0)

            groupset.append(instance)

        for indx, sam1 in enumerate(groupset):
            for sam2 in groupset[indx + 1:]:
                onto_len = len(sam1.onto.intersection(sam2.onto))
                #frame_len = len(sam1.frame.intersection(sam2.frame))
                #cosine_sim = cosine(sam1.vector, sam2.vector)
                onto_sim = 2 if onto_len == len(sam1.onto) else 1 if onto_len >= 1 else 0
                #frame_sim = 2 if frame_len == len(sam1.frame) else 1 if frame_len >= 1 else 0
                #score_dis = np.abs(sam1.score - sam2.score)
                same_main = 1 if sam1.main_sense == sam2.main_sense else 0
                both_fig = get_fig_value(sam1.figurative, sam2.figurative)
                label = 1 if sam1.cor == sam2.cor else 0
                pred = get_prediction(same_main, both_fig, sam1.onto, sam2.onto)
                homnr = group.homnr.iloc[0]

                dataset.append([sam1.lemma, sam1.ordklasse, homnr, sam1.ddo_bet, sam2.ddo_bet,
                                onto_sim, same_main, both_fig, label, pred])

    return pd.DataFrame(dataset, columns=['lemma', 'ordklasse', 'homnr',
                'bet_1', 'bet_2',
                'onto', 'main_sense',
                'figurative', 'label', 'pred'])

def get_prediction(mains, figs, onto1, onto2):
    if len(onto1) < 1 or len(onto2) < 1:
        onto = 1
    else:
        onto_length = len(onto1.intersection(onto2))
        onto = 1 if onto_length == len(onto1) else 0

    return 1 if mains == 1 and figs == 0 and onto == 1 else 0

infos = ['def']#, 'genprox', 'citat', 'kollokation']
#
# for index, info_type in enumerate(infos):
#     dataset = create_dataset(anno, infotypes=[info_type])
#     dataset.to_csv(f'var/feature_dataset_{info_type}.tsv', sep='\t', encoding='utf8')
#
#     for info_type2 in infos[index + 1:]:
#         dataset = create_dataset(anno, infotypes=[info_type, info_type2])
#         dataset.to_csv(f'var/feature_dataset_{info_type}_{info_type2}.tsv', sep='\t', encoding='utf8')
#
#         for info_type3 in infos[index + 2:]:
#             dataset = create_dataset(anno, infotypes=[info_type, info_type2, info_type3])
#             dataset.to_csv(f'var/feature_dataset_{info_type}_{info_type2}_{info_type3}.tsv', sep='\t', encoding='utf8')
#
dataset = create_dataset(anno, infotypes=infos)
dataset['acc'] = dataset.apply(lambda x: 1 if x.label == x.pred else 0, axis=1)
print(dataset['acc'].mean())

print('all')
splits = pd.read_csv(r'C:\Users\nmp828\Documents\pycor\data\splits.txt', sep='\t')
dataset_10_1 = dataset[dataset['lemma'].isin(splits['all_10_1'])]
print(dataset_10_1['acc'].mean())
dataset_10_1.to_csv('var/feature_dataset_pred_10_1.tsv', sep='\t', encoding='utf8')
dataset_10_2 = dataset[dataset['lemma'].isin(splits['all_10_2'])]
print(dataset_10_2['acc'].mean())
dataset_10_2.to_csv('var/feature_dataset_pred_10_2.tsv', sep='\t', encoding='utf8')
dataset.to_csv('var/feature_dataset_pred.tsv', sep='\t', encoding='utf8')

print('SB acc')
dataset_10_1_sb = dataset_10_1[dataset_10_1['ordklasse'] == 'sb.']
dataset_10_1_sb.to_csv('var/feature_dataset_pred_10_1_sb.tsv', sep='\t', encoding='utf8')
print(dataset_10_1_sb['acc'].mean())
dataset_10_2_sb = dataset_10_2[dataset_10_2['ordklasse'] == 'sb.']
dataset_10_2_sb.to_csv('var/feature_dataset_pred_10_2_sb.tsv', sep='\t', encoding='utf8')
print(dataset_10_2_sb['acc'].mean())

print('VB acc')
dataset_10_1_vb = dataset_10_1[dataset_10_1['ordklasse'] == 'vb.']
dataset_10_1_vb.to_csv('var/feature_dataset_pred_10_1_vb.tsv', sep='\t', encoding='utf8')
print(dataset_10_1_vb['acc'].mean())
dataset_10_2_vb = dataset_10_2[dataset_10_2['ordklasse'] == 'vb.']
dataset_10_2_vb.to_csv('var/feature_dataset_pred_10_2_vb.tsv', sep='\t', encoding='utf8')
print(dataset_10_2_vb['acc'].mean())

print('ADJ acc')
dataset_10_1_adj = dataset_10_1[dataset_10_1['ordklasse'] == 'adj.']
dataset_10_1_adj.to_csv('var/feature_dataset_pred_10_1_adj.tsv', sep='\t', encoding='utf8')
print(dataset_10_1_adj['acc'].mean())
dataset_10_2_adj = dataset_10_2[dataset_10_2['ordklasse'] == 'adj.']
dataset_10_2_adj.to_csv('var/feature_dataset_pred_10_2_adj.tsv', sep='\t', encoding='utf8')
print(dataset_10_2_adj['acc'].mean())