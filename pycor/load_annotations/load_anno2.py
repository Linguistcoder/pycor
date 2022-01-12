import pandas as pd

# columns = [score, ordklasse, lemma, definition, genprox, kollokation, hyper, ddo_bet, ddo_dannetsemid, ddo_entryid, dn_id]
#anno = pd.read_csv('data/hum_anno/text_data.txt',
#                   sep='\t',
#                   encoding='utf-8',
#                   na_values=['n', ' '])


anno = pd.read_csv('data/hum_anno/15_12_2021.txt',
                   sep='\t',
                   encoding='utf-8',
                   na_values=['n', ' '])

citater = pd.read_csv('data/citat/citater_ren.tsv',
                      sep='\t',
                      encoding='utf-8',
                      usecols=['ddo_dannetsemid', 'citat'])


anno = anno.dropna(subset=['ddo_lemma', 'COR-bet.inventar', 'ddo_dannetsemid'])
anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')

citater = citater.groupby('ddo_dannetsemid').aggregate(' '.join)

anno = anno.merge(citater, how='outer', on=['ddo_dannetsemid'])
anno = anno.dropna(subset=['ddo_lemma', 'COR-bet.inventar', 'ddo_dannetsemid', 'ddo_betyd_nr'])

anno.ddo_konbet = anno.ddo_konbet.fillna('').astype(str)
anno.ddo_encykl = anno.ddo_encykl.fillna('').astype(str)
anno.ddo_definition = anno.ddo_definition.fillna('').astype(str)

anno.ddo_definition = anno[['ddo_definition', 'ddo_konbet', 'ddo_encykl']].aggregate(' '.join, axis=1)


anno = anno.loc[:, ['score', 'ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
                    'dn_hyper', 'COR-bet.inventar', 'dn_id', 'ddo_betyd_nr', 'citat']]

anno.columns = ['score', 'lemma', 'ordklasse', 'homnr', 'definition', 'genprox', 'kollokation',
                'hyper', 'cor', 'dn_id', 'ddo_bet', 'citat']


anno = anno.dropna(subset=['score', 'lemma'])
anno = anno[anno.ordklasse.isin(['sb.', 'adj.', 'vb.'])]
anno['score'] = pd.to_numeric(anno['score'])

lemma_groups = anno.groupby(['lemma', 'ordklasse'])
unique_lemmas = anno['lemma'].unique()


# data = [ddo_sense, embedding, words, length, most_similar, lemma, genprox]

#for lemma in unique_lemmas:
