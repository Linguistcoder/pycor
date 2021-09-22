from pycor.models.word2vec import word2vec_tokenizer, word2vec_embed

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