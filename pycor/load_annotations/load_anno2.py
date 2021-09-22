import pandas as pd

# columns = [score, ordklasse, lemma, definition, genprox, kollokation, hyper, ddo_bet, ddo_dannetsemid, ddo_entryid, dn_id]
#anno = pd.read_csv('data/hum_anno/text_data.txt',
#                   sep='\t',
#                   encoding='utf-8',
#                   na_values=['n', ' '])


anno = pd.read_csv('data/hum_anno/13_09_2021.txt',
                   sep='\t',
                   encoding='utf-8',
                   na_values=['n', ' '])
anno = anno.dropna(subset=['score', 'ddo_lemma', 'COR-bet.inventar'])

citater = pd.read_csv('H:/CST_COR/data/ddo_citater_cbc.csv',
                      sep='\t',
                      encoding='utf-8')
citater = citater.groupby('ddo_dannetsemid').aggregate(' '.join)

anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')
anno = anno.merge(citater, how='outer', on='ddo_dannetsemid')
anno = anno.dropna(subset=['score', 'ddo_lemma', 'COR-bet.inventar'])

anno = anno.loc[:, ['score', 'ddo_lemma', 'ddo_ordklasse', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
                    'dn_hyper', 'COR-bet.inventar', 'dn_id', 'ddo_betyd_nr', 'citat']]

anno.columns = ['score', 'lemma', 'ordklasse', 'definition', 'genprox', 'kollokation',
                'hyper', 'cor', 'dn_id', 'ddo_bet', 'citat']


anno = anno.dropna(subset=['score', 'lemma'])
anno = anno[anno.ordklasse.isin(['sb.', 'adj.', 'vb.'])]
anno['score'] = pd.to_numeric(anno['score'])

lemma_groups = anno.groupby(['lemma', 'ordklasse'])
unique_lemmas = anno['lemma'].unique()


# data = [ddo_sense, embedding, words, length, most_similar, lemma, genprox]

#for lemma in unique_lemmas:
