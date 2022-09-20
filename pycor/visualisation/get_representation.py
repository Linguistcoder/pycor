import pandas as pd
import numpy as np
import re
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from pycor.load_annotations.load_annotations import read_anno, read_procssed_anno
from pycor.utils.preprocess import clean_wcl, get_main_sense


def reduce_dim(X, n_dim=2):
    X = normalize(X)
    pca = PCA(n_components=n_dim)
    pca.fit(X)
    transformed = pca.transform(X)
    return

def reduce_dim2(X, n_dim=2):
    fit = umap.UMAP(n_components=n_dim, n_neighbors=5, min_dist=0.2)
    fit.fit(X)
    return fit.transform(X)



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
        self.data = data.groupby(['ddo_lemma', 'ddo_homnr'])

    @staticmethod
    def get_representation(row, vector, max_sense):
        if isinstance(vector, str):
            vector = re.sub('\[ +', '[', vector)
            vector = re.sub('[ \n]+', ', ', vector)
            vector = np.array(eval(vector))

        return {'sense': row.ddo_betyd_nr,
                'cor': int(row.cor_bet_inventar) + int(max_sense),
                'embedding': vector,
                'length': len(row.bow),
                # 'most_similar': word2vec.most_similar(positive=[vector], topn=n_sim) if n_sim else '',
                'lemma': row.ddo_lemma,
                'genprox': row.ddo_genprox,
                'score': row.score,
                'definition': row.ddo_definition}

    def get_representation_for_lemmas(self, lemmas, model_name):
        data = []
        extra_lemmas = []
        add_n = 0
        for index, (lemma, homnr) in enumerate(lemmas):
            group = self.data.get_group((lemma, homnr))
            add_n += int(group.cor_bet_inventar.max())

            for row in group.itertuples():
                vector = row[group.columns.get_loc(model_name)+1]
                content = self.get_representation(row, vector, add_n)
                content['index'] = index + 1
                data.append(content)
                extra_lemmas.append(row.ddo_genprox)
                stikord = row.cor_stikord.split(',')
                if len(stikord):
                    extra_lemmas += stikord

            extra_lemmas = list(set(extra_lemmas))

        for index, lemma in enumerate(extra_lemmas):
            for i in range(1, 5):
                if (lemma, i) in self.data.groups:
                    group = self.data.get_group((lemma, i))
                    for row in group.itertuples():
                        vector = row[group.columns.get_loc(model_name) + 1]
                        content = self.get_representation(row, vector, 0)
                        content['cor'] = add_n + 2
                        content['score'] = 1
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
        senses_2dim = reduce_dim2(np.vstack([a for i, a in dataset.embedding.items()]))

        labels = [(f"DDO_sense:{row.sense}<br>" +
                   f"COR: {row.cor}<br>" +
                   f"Lemma: {row.lemma}<br>" +
                   f"GenProx: {row.genprox}<br>" 
                   f"Definition: {row.definition}<br>" +
                   f"n_words {row.length}<br>')") for row in dataset.itertuples()]

        return senses_2dim, list(dataset['cor']), labels, list(dataset['score']), list(dataset['index'])
