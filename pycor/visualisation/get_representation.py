import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from pycor.load_annotations.load_anno2 import lemma_groups as ANNO
from pycor.models.word2vec import word2vec_embed, word2vec_tokenizer, word2vec
from pycor.utils.preprocess import clean_wcl, get_main_sense
from pycor.utils.vectors import vectorize


def reduce_dim(X, n_dim=2):
    pca = PCA(n_components=n_dim)
    pca.fit(X)
    transformed = pca.transform(X)
    return transformed


def get_representation_for_lemma(lemma, wcl, lemma_groups, n_sim=1):

    group = lemma_groups.get_group((lemma, clean_wcl(wcl)))

    data = []
    for row in group.itertuples():
        vector = vectorize(row, infotypes=['def', 'citat'])
        if type(vector) != float:
            data.append({'sense': row.ddo_bet,
                         'embedding': vector,
                         #'words': word2vec_tokenizer(row.definition),
                         'length': len(word2vec_tokenizer(row.definition)),
                         'most_similar': word2vec.most_similar(positive=[vector], topn=n_sim) if n_sim else '',
                         'lemma': row.lemma,
                         'genprox': row.genprox,
                         'score': row.score})

    return pd.DataFrame(data)


def get_2d_representation_from_lemma(lemma, wcl, n_sim=1, lemma_groups=ANNO, include=None):
    if include is None:
        include = ['lemma', 'genprox']

    data: pd.DataFrame = get_representation_for_lemma(lemma, wcl, lemma_groups, n_sim)

    # if lemma == 'stang':
    #     for s in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', ]:
    #         mean_embedding = [row.embedding for row in data.itertuples() if s in row.sense and row.sense != '2.c']
    #         mean_embedding2 = np.mean(mean_embedding, axis=0)
    #         data = data.append({'sense': s,
    #                                     'embedding': mean_embedding2,
    #                                     #'words': s,
    #                                     'length': len(mean_embedding),
    #                                     'most_similar': np.nan,
    #                                     'lemma': f'{lemma}',
    #                                     'genprox': np.nan,
    #                                     'score': 1}, ignore_index=True)

    if include:
        for column_name in include:
            column = data[column_name]
            for index, wordform in column.items():
                if not type(wordform) == str:
                    continue
                else:
                    data = data.append({'sense': 'word2vec-' + wordform,
                                        'embedding': word2vec_embed(wordform),
                                        #'words': 'word2vec-' + wordform,
                                        'length': 1,
                                        'most_similar': np.nan,
                                        'lemma': f'{lemma}_{str(index+1)}',
                                        'genprox': np.nan,
                                        'score': 0}, ignore_index=True)

    if n_sim:
        for row in data.itertuples():
            if type(row.most_similar) == float:
                continue

            for word, sim in row.most_similar:
                data = data.append({'sense': word,
                                    'embedding': word2vec_embed(word),
                                    #'words': 'similar-to-' + row.sense,
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
               #f"Words: {row.words}<br>" +
               f"n_words {row.length}<br>')") for row in data.itertuples()]

    ddo_sense = [get_main_sense(row.sense) for row in data.itertuples()]
    return senses_2dim, list(data['sense']), labels, list(data['score']), ddo_sense
