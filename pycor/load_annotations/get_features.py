import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from pycor.load_annotations.datasets import DataInstance
from pycor.utils.preprocess import clean_ontology, clean_frame, get_fig_value
from pycor.utils.vectors import vectorize

anno = pd.read_csv('data/hum_anno/17_09_2021.txt',
                   sep='\t',
                   encoding='utf-8',
                   na_values=['n', ' '])

citater = pd.read_csv('H:/CST_COR/data/ddo_citater_cbc.csv',
                      sep='\t',
                      encoding='utf-8')
citater = citater.groupby('ddo_dannetsemid').aggregate(' '.join)

# columns = ['score', 'ddo_entryid', 'ddo_lemma', 'ddo_homnr', 'ddo_ordklasse',
#        'ddo_dannetsemid', 'ddo_definition', 'ddo_genprox', 'ddo_bemaerk',
#        'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame', 'ddo_valens',
#        'ddo_kollokation', 'COR-bet.inventar', 'Hvorfor?', 'Ny ordklasse',
#        'ddo_betyd_nr', 'ddo_bet_tags', 'ddo_senselevel', 'ddo_plac',
#        'ddo_art_tags', 'ddo_bet', 'ddo_mwe_bet',
#        'Antal citater til bet/ddo_bet_doks', 'ddo_art_doks', 'ddo_udvalg',
#        'dn_id', 'dn_lemma', 'ddo_sublemma', 'ddb_nøgleord']

anno = anno.dropna(subset=['score', 'ddo_lemma', 'COR-bet.inventar'])
anno = anno[anno['COR-bet.inventar'] != '0']

anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')
anno = anno.merge(citater, how='outer', on='ddo_dannetsemid')
anno = anno.dropna(subset=['score', 'ddo_lemma', 'COR-bet.inventar'])

anno = anno.loc[:, ['ddo_lemma', 'ddo_ordklasse', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
                    'COR-bet.inventar', 'dn_id', 'ddo_betyd_nr', 'citat', 'score', 'ddo_bemaerk',
                    'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame']]

anno.columns = ['lemma', 'ordklasse', 'definition', 'genprox', 'kollokation', 'cor', 'dn_id', 'ddo_nr', 'citat',
                't_score', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame']


# DanNet = load_obj('DanNet')
# DanNet = expand_synsets(expand_synsets(DanNet), s=False)


def create_dataset(annotations, infotypes=['def']):
    dataset = [['lemma', 'ordklasse',
                'bet_1', 'bet_2', 'cosine',
                'onto', 'frame', 'score', 'main_sense',
                'figurative', 'label']]

    for name, group in annotations.groupby(['lemma', 'ordklasse']):
        groupset = []
        #if len(group.index) > 5:
            #continue
        for row in group.itertuples():
            figurative = row.bemaerk if type(row.bemaerk) != float else ''
            instance = DataInstance(lemma=row.lemma,
                                    wcl=row.ordklasse,
                                    cor=row.cor,
                                    ddo_bet=row.ddo_nr,
                                    vector=vectorize(row, infotypes=infotypes),
                                    onto=clean_ontology(row.onto1).union(clean_ontology(row.onto2)),
                                    frame=clean_frame(row.frame),
                                    score=int(row.t_score),
                                    figurative=1 if 'ofø' in figurative else 0)

            groupset.append(instance)

        for indx, sam1 in enumerate(groupset):
            for sam2 in groupset[indx + 1:]:
                onto_len = len(sam1.onto.intersection(sam2.onto))
                frame_len = len(sam1.frame.intersection(sam2.frame))

                cosine_sim = cosine(sam1.vector, sam2.vector)
                onto_sim = 2 if onto_len == len(sam1.onto) else 1 if onto_len >= 1 else 0
                frame_sim = 2 if frame_len == len(sam1.frame) else 1 if frame_len >= 1 else 0
                score_dis = np.abs(sam1.score - sam2.score)
                same_main = 1 if sam1.main_sense == sam2.main_sense else 0
                both_fig = get_fig_value(sam1.figurative, sam2.figurative)
                label = 1 if sam1.cor == sam2.cor else 0

                dataset.append([sam1.lemma, sam1.wcl, sam1.ddo_bet, sam2.ddo_bet,
                                cosine_sim, onto_sim, frame_sim, score_dis,
                                same_main, both_fig, label])

    return pd.DataFrame(dataset)


infos = ['def', 'genprox', 'citat', 'kollokation']

for index, info_type in enumerate(infos):
    dataset = create_dataset(anno, infotypes=[info_type])
    dataset.to_csv(f'var/feature_dataset_{info_type}.tsv', sep='\t', encoding='utf8')

    for info_type2 in infos[index + 1:]:
        dataset = create_dataset(anno, infotypes=[info_type, info_type2])
        dataset.to_csv(f'var/feature_dataset_{info_type}_{info_type2}.tsv', sep='\t', encoding='utf8')

        for info_type3 in infos[index + 2:]:
            dataset = create_dataset(anno, infotypes=[info_type, info_type2, info_type3])
            dataset.to_csv(f'var/feature_dataset_{info_type}_{info_type2}_{info_type3}.tsv', sep='\t', encoding='utf8')

    dataset = create_dataset(anno, infotypes=infos)
    dataset.to_csv('var/feature_dataset_all.tsv', sep='\t', encoding='utf8')
