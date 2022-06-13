import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from pycor.models.word2vec import word2vec_embed, word2vec_tokenizer, word2vec
from pycor.utils.preprocess import clean_wcl, get_main_sense
from pycor.utils.vectors import vectorize


def reduce_dim(X, n_dim=2):
    pca = PCA(n_components=n_dim)
    pca.fit(X)
    transformed = pca.transform(X)
    return transformed


def get_w2v_representation_for_lemma(lemma, wcl, data, n_sim=1):
    lemma_groups = data.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr'])
    group = lemma_groups.get_group((lemma, clean_wcl(wcl)))

    data = []
    for row in group.itertuples():
        vector = vectorize(row.bow)
        if type(vector) != float:
            data.append({'sense': row.ddo_bet_nr,
                         'cor': row.COR_bet_inventar,
                         'embedding': vector,
                         'length': len(word2vec_tokenizer(row.definition)),
                         'most_similar': word2vec.most_similar(positive=[vector], topn=n_sim) if n_sim else '',
                         'lemma': row.ddo_lemma,
                         'genprox': row.ddo_genprox,
                         'score': row.score})

    return pd.DataFrame(data)


def get_2d_bert_representation_from_lemma(lemma, wcl, data):
    lemma_groups = data.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr'])

    group = lemma_groups.get_group((lemma, clean_wcl(wcl)))

    data = []
    for row in group.itertuples():
        data.append({'sense': row.ddo_bet_nr,
                     'cor': row.COR_bet_inventar,
                     'embedding': row.embedding,
                     'length': len(word2vec_tokenizer(row.definition)),
                     'lemma': row.ddo_lemma,
                     'genprox': row.ddo_genprox,
                     'score': row.score})

    return pd.DataFrame(data)



def get_2d_w2v_representation_from_lemma(lemma, wcl, data, n_sim=1):
    lemma_groups = data.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr'])


    dataset: pd.DataFrame = get_w2v_representation_for_lemma(lemma, wcl, lemma_groups, n_sim)

    for wordform in ['lemma', 'genprox']:
        if not type(wordform) == str:
            continue
        else:
            dataset = dataset.append({'sense': 'word2vec-' + wordform,
                                      'embedding': word2vec_embed(wordform),
                                      'length': 1,
                                      'most_similar': np.nan,
                                      'lemma': f'{lemma}_word2vec',
                                      'genprox': np.nan,
                                      'score': 0}, ignore_index=True)

    if n_sim:
        for row in data.itertuples():
            if type(row.most_similar) == float:
                continue

            for word, sim in row.most_similar:
                data = data.append({'sense': word,
                                    'embedding': word2vec_embed(word),
                                    'length': 1,
                                    'most_similar': np.nan,
                                    'lemma': 'similar-to-' + row.sense,
                                    'genprox': np.nan,
                                    'score': 0}, ignore_index=True)

    data = data.dropna(subset=['embedding'])
    senses_2dim = reduce_dim(np.vstack([a for i, a in data.embedding.items()]))

    labels = [(f"DDO_sense:{row.sense}<br>" +
               f"Lemma: {row.lemma}<br>" +
               f"GenProx: {row.genprox}<br>" +
               f"n_words {row.length}<br>')") for row in data.itertuples()]

    ddo_sense = [get_main_sense(row.sense) for row in data.itertuples()]
    return senses_2dim, list(data['sense']), labels, list(data['score']), ddo_sense
