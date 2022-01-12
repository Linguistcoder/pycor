import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from pycor.DanNet.dan_utils import expand_synsets
from pycor.utils.save_load import load_obj
from pycor.utils.vectors import vectorize

anno = pd.read_csv('data/hum_anno/13_09_2021.txt',
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
anno = anno[anno['COR-bet.inventar']!='0']

anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')
anno = anno.merge(citater, how='outer', on='ddo_dannetsemid')
anno = anno.dropna(subset=['score', 'ddo_lemma', 'COR-bet.inventar'])


anno = anno.loc[:, ['ddo_lemma', 'ddo_ordklasse', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
                    'COR-bet.inventar', 'dn_id', 'ddo_betyd_nr', 'citat']]

anno.columns = ['lemma', 'ordklasse', 'definition', 'genprox', 'kollokation', 'cor', 'dn_id', 'ddo_nr', 'citat']

DanNet = load_obj('DanNet')
DanNet = expand_synsets(expand_synsets(DanNet), s=False)


def create_dataset(annotations, infotypes=['def']):
    dataset = [['lemma','bet_1', 'bet_2', 'score', 'label']]
    for name, group in annotations.groupby(['lemma', 'ordklasse']):
        groupset = []
        #if len(group.index) > 5:
            #continue
        for row in group.itertuples():
            vector = vectorize(row, infotypes=infotypes)
            groupset.append((vector, row.cor, row.lemma, row.ddo_nr))

        for index, vector in enumerate(groupset):
            for vec, cor_id, l, nr in groupset[index+1:]:
                sim = cosine(vector[0], vec)
                label = 0 if vector[1] == cor_id else 1

                dataset.append([vector[2], vector[3], nr, sim, label])

    return pd.DataFrame(dataset)

infos = ['def', 'genprox', 'citat', 'kollokation']

for index, info_type in enumerate(infos):
    dataset = create_dataset(anno, infotypes=[info_type])
    dataset.to_csv(f'var/cosine_dataset_{info_type}.tsv', sep='\t', encoding='utf8')

    for info_type2 in infos[index+1:]:
        dataset = create_dataset(anno, infotypes=[info_type, info_type2])
        dataset.to_csv(f'var/cosine_dataset_{info_type}_{info_type2}.tsv', sep='\t', encoding='utf8')

        for info_type3 in infos[index+2:]:
            dataset = create_dataset(anno, infotypes=[info_type, info_type2, info_type3])
            dataset.to_csv(f'var/cosine_dataset_{info_type}_{info_type2}_{info_type3}.tsv', sep='\t', encoding='utf8')


    dataset = create_dataset(anno, infotypes=infos)
    dataset.to_csv('var/_cosine_dataset_all.tsv', sep='\t', encoding='utf8')
