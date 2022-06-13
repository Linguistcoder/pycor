from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score

from pycor.load_annotations.load_annotations import read_procssed_anno, read_anno
from pycor.utils.preprocess import clean_ontology, clean_frame, get_fig_value, get_main_sense

#from pycor.utils.vectors import vectorize

# anno = pd.read_csv('data/hum_anno/mellem_07_03_2022.txt',
#                    sep='\t',
#                    encoding='utf-8',
#                    na_values=['n', ' '],
#                    index_col=False
#                    )
#
# citater = pd.read_csv('data/citat/citater_ren.tsv',
#                       sep='\t',
#                       encoding='utf-8',
#                       usecols=['ddo_dannetsemid', 'citat'])

# anno = pd.read_csv('data/hum_anno/mellemfrekvente.txt',
#                    sep='\t',
#                    encoding='utf-8',
#                    na_values=['n', ' '])
#
# citater = pd.read_csv('H:/CST_COR/data/DDO_citater/citater_mellemfrekvente.tsv',
#                       sep='\t',
#                       encoding='utf-8')
# citater = citater.groupby('ddo_dannetsemid').aggregate(' '.join)

# columns = ['score', 'ddo_entryid', 'ddo_lemma', 'ddo_homnr', 'ddo_ordklasse',
#        'ddo_dannetsemid', 'ddo_definition', 'ddo_genprox', 'ddo_bemaerk',
#        'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame', 'ddo_valens',
#        'ddo_kollokation', 'COR-bet.inventar', 'Hvorfor?', 'Ny ordklasse',
#        'ddo_betyd_nr', 'ddo_bet_tags', 'ddo_senselevel', 'ddo_plac',
#        'ddo_art_tags', 'ddo_bet', 'ddo_mwe_bet',
#        'Antal citater til bet/ddo_bet_doks', 'ddo_art_doks', 'ddo_udvalg',
#        'dn_id', 'dn_lemma', 'ddo_sublemma', 'ddb_nøgleord']
#
# anno = anno.dropna(subset=['ddo_lemma', 'COR_bet_inventar', 'ddo_dannetsemid', 'ddo_betyd_nr'])
# anno = anno[anno['COR_bet_inventar'] != '0']
#
# anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')
# anno = anno.merge(citater, how='outer', on='ddo_dannetsemid')
# anno = anno.dropna(subset=['ddo_lemma', 'COR_bet_inventar'])
#
#
# anno = anno.loc[:, ['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
#                     'COR_bet_inventar', 'dn_id', 'ddo_betyd_nr', 'citat', 'ddo_bemaerk',
#                     'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame', 'ddo_konbet', 'ddo_encykl']]
#
# anno.columns = ['lemma', 'ordklasse', 'homnr', 'definition', 'genprox', 'kollokation', 'cor', 'dn_id', 'ddo_nr',
#                 'citat', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame', 'konbet', 'encykl']
#
# # DanNet = load_obj('DanNet')
# # DanNet = expand_synsets(expand_synsets(DanNet), s=False)
DataInstance = namedtuple('DataInstance', ['lemma', 'ordklasse', 'homnr', 'ddo_bet', 'main_sense', 'cor',
                           'onto', 'figurative'])

def create_dataset(annotations, words='all', limit=0):
    dataset = []
    annotations['ddo_homnr'] = annotations['ddo_homnr'].fillna(1)
    annotations = annotations[annotations['COR_bet_inventar'] != 0]

    for name, group in annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']):
        groupset = []

        wcl = name[1].lower()
        if words != 'all' and words not in wcl:
            continue

        if 0 < limit <= len(group.index):
            continue

        #print(name[0].lower())

        for row in group.itertuples():
            figurative = row.ddo_bemaerk if type(row.ddo_bemaerk) != float else ''
            instance = DataInstance(lemma=row.ddo_lemma,
                                    ordklasse=row.ddo_ordklasse,
                                    homnr=row.ddo_homnr,
                                    ddo_bet=row.ddo_betyd_nr,
                                    main_sense = get_main_sense(row.ddo_betyd_nr),
                                    cor=row.COR_bet_inventar,
                                    onto=clean_ontology(row.dn_onto1).union(clean_ontology(row.dn_onto2)),
                                    figurative=1 if 'ofø' in figurative else 0)

            groupset.append(instance)

        for indx, sam1 in enumerate(groupset):
            for sam2 in groupset[indx + 1:]:
                onto_len = len(sam1.onto.intersection(sam2.onto))
                #frame_len = len(sam1.frame.intersection(sam2.frame))
                #cosine_sim = cosine(sam1.vector, sam2.vector)
                onto_sim = 1 if onto_len == len(sam1.onto) else 0.5 if onto_len >= 1 else 0
                #frame_sim = 2 if frame_len == len(sam1.frame) else 1 if frame_len >= 1 else 0
                #score_dis = np.abs(sam1.score - sam2.score)
                same_main = 1 if sam1.main_sense == sam2.main_sense else 0
                both_fig = get_fig_value(sam1.figurative, sam2.figurative)
                label = 1 if sam1.cor == sam2.cor else 0
                pred = get_prediction(same_main, both_fig, sam1.onto, sam2.onto)
                homnr = group.ddo_homnr.iloc[0]

                dataset.append([sam1.lemma, sam1.ordklasse, homnr, sam1.ddo_bet, sam2.ddo_bet,
                                onto_sim, same_main, both_fig, label, pred])

    return pd.DataFrame(dataset, columns=['lemma', 'ordklasse', 'homnr',
                'bet_1', 'bet_2',
                'onto', 'main_sense',
                'figurative', 'label', 'score'])

def get_prediction(mains, figs, onto1, onto2):
    if len(onto1) < 1 or len(onto2) < 1:
        onto = 1
    else:
        onto_sim = onto1.intersection(onto2)
        onto_length = len(onto1.intersection(onto2))
        #onto = 1 if onto_length == len(onto1) else 0
        onto = 1 if onto_sim == onto1 or onto_sim == onto2 else 0

    return 1 if mains == 1 and figs == 0 and onto == 1 else 0



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
#datasets = ['cbc_train', 'cbc_devel', 'cbc_test', 'keywords_train', 'keywords_test', 'mellem_train', 'mellem_test']

#datasets = ['cbc_train', 'cbc_devel', 'cbc_test', 'keywords_train', 'keywords_test']
datasets = ['hum_anno/CBC_valid_fini.txt']

for subset in datasets:
    anno = read_procssed_anno(f'data/{subset}')#2.tsv')

    wordclass = 'all'
    dataset = create_dataset(anno, words=wordclass, limit=5)

    dataset['acc'] = dataset.apply(lambda x: 1 if x.label == x.score else 0, axis=1)
    #dataset.to_csv(f'data/base/reduction_score_{subset}2.tsv', sep='\t', encoding='utf8')
    dataset.to_csv(f'data/base/reduction_score_CBC_valid.tsv', sep='\t', encoding='utf8')
    #dataset.to_csv(f'data/base/reduction_score_{subset}2_{wordclass}.tsv', sep='\t', encoding='utf8')
    #dataset.to_csv(f'data/base/reduction_score_{subset}2_less.tsv', sep='\t', encoding='utf8')

    mis = dataset[dataset['acc'] == 0]

    print(f'_________________ {subset.upper()}_________________________________')
    print(len(dataset))
    print(dataset['acc'].mean())
    print('F1 all:', f1_score(dataset['score'], dataset['label']))
#
# # print('all')
# splits = pd.read_csv(r'C:\Users\nmp828\Documents\pycor\data\splits.txt', sep='\t')
# dataset_10_1 = dataset[dataset['lemma'].isin(splits['all_10_1'])]
# print(dataset_10_1['acc'].mean())
# print('F1 10_1:', f1_score(dataset_10_1['pred'], dataset_10_1['label']))
#
# dataset_10_1.to_csv('var/feature_dataset_pred_10_1.tsv', sep='\t', encoding='utf8')
# dataset_10_2 = dataset[dataset['lemma'].isin(splits['all_10_2'])]
# print(dataset_10_2['acc'].mean())
# print('F1 10_2:', f1_score(dataset_10_2['pred'], dataset_10_2['label']))
# dataset_10_2.to_csv('var/feature_dataset_pred_10_2.tsv', sep='\t', encoding='utf8')
# #dataset.to_csv('var/feature_dataset_pred.tsv', sep='\t', encoding='utf8')
# #
# print('SB acc')
# dataset_10_1_sb = dataset_10_1[dataset_10_1['ordklasse'] == 'sb.']
# dataset_10_1_sb.to_csv('var/feature_dataset_pred_10_1_sb.tsv', sep='\t', encoding='utf8')
# print(dataset_10_1_sb['acc'].mean())
# print('F1 10_1_sb:', f1_score(dataset_10_1_sb['pred'], dataset_10_1_sb['label']))
#
# dataset_10_2_sb = dataset_10_2[dataset_10_2['ordklasse'] == 'sb.']
# dataset_10_2_sb.to_csv('var/feature_dataset_pred_10_2_sb.tsv', sep='\t', encoding='utf8')
# print(dataset_10_2_sb['acc'].mean())
# print('F1 10_2_sb:', f1_score(dataset_10_2_sb['pred'], dataset_10_2_sb['label']))
#
# print('VB acc')
# dataset_10_1_vb = dataset_10_1[dataset_10_1['ordklasse'] == 'vb.']
# dataset_10_1_vb.to_csv('var/feature_dataset_pred_10_1_vb.tsv', sep='\t', encoding='utf8')
# print(dataset_10_1_vb['acc'].mean())
# print('F1 10_1_vb:', f1_score(dataset_10_1_vb['pred'], dataset_10_1_vb['label']))
#
# dataset_10_2_vb = dataset_10_2[dataset_10_2['ordklasse'] == 'vb.']
# dataset_10_2_vb.to_csv('var/feature_dataset_pred_10_2_vb.tsv', sep='\t', encoding='utf8')
# print(dataset_10_2_vb['acc'].mean())
# print('F1 10_2_vb:', f1_score(dataset_10_2_vb['pred'], dataset_10_2_vb['label']))
#
# print('ADJ acc')
# dataset_10_1_adj = dataset_10_1[dataset_10_1['ordklasse'] == 'adj.']
# dataset_10_1_adj.to_csv('var/feature_dataset_pred_10_1_adj.tsv', sep='\t', encoding='utf8')
# print(dataset_10_1_adj['acc'].mean())
# print('F1 10_1_adj:', f1_score(dataset_10_1_adj['pred'], dataset_10_1_adj['label']))
# dataset_10_2_adj = dataset_10_2[dataset_10_2['ordklasse'] == 'adj.']
# dataset_10_2_adj.to_csv('var/feature_dataset_pred_10_2_adj.tsv', sep='\t', encoding='utf8')
# print(dataset_10_2_adj['acc'].mean())
# print('F1 10_2_adj:', f1_score(dataset_10_2_adj['pred'], dataset_10_2_adj['label']))