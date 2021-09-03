import re

from pycor.models.word2vec import word2vec_embed, word2vec_tokenizer
from pycor.utils.save_load import load_obj
from pycor.DanNet.dan_utils import expand_synsets

DanNet = load_obj('DanNet')
DanNet = expand_synsets(expand_synsets(DanNet), s=False)


def vectorize(row):
    sentence = word2vec_tokenizer(row.definition)
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
    return vector


def get_main_sense(sense):
    if any(char.isdigit() for char in sense) and 'word2vec' not in sense:
        sense = re.sub('[^0-9]', '', sense)
        if sense == '':
            return 0
        return int(sense) * 2
    else:
        return 0


def clean_wcl(wcl):
    if 'sb' in wcl or 'sub' in wcl:
        return 'sb.'
    elif 'vb' in wcl or 'ver' in wcl:
        return 'vb.'
    elif 'adj' in wcl:
        return 'adj.'
