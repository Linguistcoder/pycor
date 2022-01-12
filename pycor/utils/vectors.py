from pycor.models.word2vec import word2vec_tokenizer, word2vec_embed
from scipy.spatial.distance import cosine
import numpy as np

def vectorize(row, infotypes=['def']):
    sentence = [row.lemma]

    if 'def' in infotypes:
        sentence += word2vec_tokenizer(row.definition)

    if 'kollokation' in infotypes:
        if row.kollokation and type(row.kollokation) != float:
            sentence += list(set(word2vec_tokenizer(row.kollokation)))

    if 'citat' in infotypes:
        if row.citat and type(row.citat) != float:
            sentence += word2vec_tokenizer(row.citat)

    if 'genprox' in infotypes:
        sentence += [row.genprox]

    if 'DanNet' in infotypes:
        raise NotImplementedError

    """
    dn_id = row.dn_id
    if dn_id and type(dn_id) != float:
        if ';' in dn_id:
            dn_id = dn_id.split(';')
            for id in dn_id:
                synset = DanNet.get(int(id), None)
                if synset:
                    sentence += synset.get_example_sentence()
        else:
            synset = DanNet.get(int(row.dn_id), None)
            if synset:
                sentence += synset.get_example_sentence()
    """

    vector = word2vec_embed(sentence)

    return vector


def vectorize_rows(row):
    vector1 = word2vec_embed(word2vec_tokenizer(row.sentence_1))
    vector2 = word2vec_embed(word2vec_tokenizer(row.sentence_2))

    concatted = np.concatenate((vector1, vector2), axis=None)

    if concatted.shape == (1000,):
        return concatted
    else:
        return np.zeros((1000,)) + 0.00001


def vectorize_and_cosine(row):
    vector1 = word2vec_embed(word2vec_tokenizer(row.sentence_1))
    vector2 = word2vec_embed(word2vec_tokenizer(row.sentence_2))

    return cosine(vector1, vector2)
