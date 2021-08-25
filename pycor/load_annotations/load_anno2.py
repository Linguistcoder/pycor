import pandas as pd


anno = pd.read_csv('data/hum_anno/text_data.txt',
                   sep='\t',
                   encoding='utf-8',
                   na_values=['n', ' '])

# columns = [score, ordklasse, lemma, definition, genprox, kollokation, hyper, ddo_bet, ddo_dannetsemid, ddo_entryid, dn_id]
anno = anno.dropna(subset=['score', 'lemma'])
anno = anno[anno.ordklasse.isin(['sb.', 'adj.', 'vb.'])]
anno['score'] = pd.to_numeric(anno['score'])

lemma_groups = anno.groupby(['lemma', 'ordklasse'])
unique_lemmas = anno['lemma'].unique()

# data = [ddo_sense, embedding, words, length, most_similar, lemma, genprox]

#for lemma in unique_lemmas:


