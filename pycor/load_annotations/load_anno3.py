import pandas as pd

from pycor.DanNet.dan_utils import expand_synsets
#from pycor.models.word2vec import word2vec_tokenizer, word2vec_embed
from pycor.utils.save_load import load_obj

anno = pd.read_csv('data/hum_anno/Fælles arbejdsfil cor_s_3.xlsx - Ark1.tsv',
                   sep='\t',
                   encoding='utf-8',
                   na_values=['n', ' '])

# columns = ['score', 'ddo_entryid', 'ddo_lemma', 'ddo_homnr', 'ddo_ordklasse',
#        'ddo_dannetsemid', 'ddo_definition', 'ddo_genprox', 'ddo_bemaerk',
#        'dn_onto1', 'dn_onto2', 'dn_hyper', 'frame', 'ddo_valens',
#        'ddo_kollokation', 'COR-bet.inventar', 'Hvorfor?', 'Ny ordklasse',
#        'ddo_betyd_nr', 'ddo_bet_tags', 'ddo_senselevel', 'ddo_plac',
#        'ddo_art_tags', 'ddo_bet', 'ddo_mwe_bet',
#        'Antal citater til bet/ddo_bet_doks', 'ddo_art_doks', 'ddo_udvalg',
#        'dn_id', 'dn_lemma', 'ddo_sublemma', 'ddb_nøgleord']

anno = anno.dropna(subset=['score', 'ddo_lemma', 'COR-bet.inventar'])

anno = anno.loc[:, ['ddo_lemma', 'ddo_ordklasse', 'ddo_definition', 'ddo_genprox', 'ddo_kollokation',
                    'COR-bet.inventar', 'dn_id']]

anno.columns = ['lemma', 'ordklasse', 'definition', 'genprox', 'kollokation', 'cor', 'dn_id']

DanNet = load_obj('DanNet')
DanNet = expand_synsets(expand_synsets(DanNet), s=False)


"""def vectorize(row):
    sentence = word2vec_tokenizer(row.definition)

    if row.kollokation:
        sentence += word2vec_tokenizer(row.kollokation)

    dn_id = row.dn_id
    if dn_id and type(dn_id) != float:
        if ';' in dn_id:
            dn_id = dn_id.split(';')
            for id in dn_id:
                sentence += DanNet.get(int(id), None).get_example_sentence()
        else:
            synset = DanNet.get(int(row.dn_id), None)
            if synset:
                sentence += synset.get_example_sentence()

    sentence += [row.lemma] + [row.genprox]
    vector = word2vec_embed(sentence)
    return vector"""

def create_dataset(annotations):
    for name, group in annotations.groupby(['lemma', 'ordklasse']):
        print(name)
        print(group)

create_dataset(anno)


