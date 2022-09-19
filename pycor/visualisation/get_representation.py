import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from pycor.load_annotations.load_annotations import read_anno, read_procssed_anno
from pycor.utils.preprocess import clean_wcl, get_main_sense


def reduce_dim(X, n_dim=2):

    X = normalize(X)
    pca = PCA(n_components=n_dim)
    pca.fit(X)
    transformed = pca.transform(X)
    return transformed


def annotations_to_embeddings(anno_file, vector_file):
    annotations = read_anno(anno_file=anno_file,
                            quote_file='',
                            keyword_file='',
                            annotated=False)

    vectors = read_procssed_anno(vector_file)

    annotations = annotations.merge(vectors, how='outer', on=['lobenummer', 'ddo_dannetsemid'])

    return Embeddings(annotations)

class Embeddings(object):
    def __init__(self, data):
        self.data = data.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr'])

    @staticmethod
    def get_representation(row, vector):
        if isinstance(vector, str):
            vector = re.sub('\[ +', '[', vector)
            vector = re.sub('[ \n]+', ', ', vector)
            vector = np.array(eval(vector))

        return {'sense': row.ddo_betyd_nr,
                'cor': row.cor_bet_inventar,
                'embedding': vector,
                'length': len(row.bow),
                # 'most_similar': word2vec.most_similar(positive=[vector], topn=n_sim) if n_sim else '',
                'lemma': row.ddo_lemma,
                'genprox': row.ddo_genprox,
                'score': row.score}

    def get_representation_for_lemmas(self, lemmas, model_name):
        data = []
        for index, (lemma, wcl, homnr) in enumerate(lemmas):
            group = self.data.get_group((lemma, clean_wcl(wcl), homnr))

            for row in group.itertuples():
                vector = row[group.columns.get_loc(model_name)+1]
                content = self.get_representation(row, vector)
                content['index'] = index + 1
                data.append(content)

        return pd.DataFrame(data)



    def get_2d_representations_from_lemmas(self, lemmas, model_name):
        dataset: pd.DataFrame = self.get_representation_for_lemmas(lemmas, model_name)

        # for wordform in ['lemma', 'genprox']:
        #     if not type(wordform) == str:
        #         continue
        #     else:
        #         dataset = dataset.append({'sense': 'word2vec-' + wordform,
        #                                   'embedding': word2vec_embed(wordform),
        #                                   'length': 1,
        #                                   'most_similar': np.nan,
        #                                   'lemma': f'{lemma}_word2vec',
        #                                   'genprox': np.nan,
        #                                   'score': 0}, ignore_index=True)

        # if n_sim:
        #     for row in data.itertuples():
        #         if type(row.most_similar) == float:
        #             continue
        #
        #         for word, sim in row.most_similar:
        #             data = data.append({'sense': word,
        #                                 'embedding': word2vec_embed(word),
        #                                 'length': 1,
        #                                 'most_similar': np.nan,
        #                                 'lemma': 'similar-to-' + row.sense,
        #                                 'genprox': np.nan,
        #                                 'score': 0}, ignore_index=True)

        dataset = dataset.dropna(subset=['embedding'])
        senses_2dim = reduce_dim(np.vstack([a for i, a in dataset.embedding.items()]))

        labels = [(f"DDO_sense:{row.sense}<br>" +
                   f"COR: {row.cor}<br>" +
                   f"Lemma: {row.lemma}<br>" +
                   f"GenProx: {row.genprox}<br>" +
                   f"n_words {row.length}<br>')") for row in dataset.itertuples()]

        return senses_2dim, list(dataset['cor']), labels, list(dataset['score']), list(dataset['index'])
