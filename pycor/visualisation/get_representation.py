import pandas as pd
import numpy as np
import re

from sklearn.metrics import pairwise_distances

from pycor.load_annotations.load_annotations import read_anno, read_procssed_anno


def annotations_to_embeddings(anno_file, vector_file, reducer):
    """
    Create a EmbeddingForVisualisation object from:
        - a file with the sense inventory (anno_file) with lexical information (dictionary data)
        - a file with BERT and word2vec embeddings for each sense in the sense inventory defined above.
        The two files are merged on the COR lobenummer and DSLs ddo_dannetsemid

        :param anno_file: (str) filename of tab-separated sense inventory file
        :param vector_file: (str) filename of tab-separated embeddings file
        :param reducer: (func) dimensionality reduction method
        :return: EmbeddingsForVisualisation object
    """

    def get_vector_from_string(vector):
        """Transforms input string into numpy array"""
        if isinstance(vector, str):
            vector = re.sub(r'\[ +', '[', vector)
            vector = re.sub(r'[ \n]+', ', ', vector)
            vector = np.array(eval(vector))
        return vector

    # read input sense inventory file with dictionary data
    annotations = read_anno(anno_file=anno_file,
                            quote_file='',
                            keyword_file='',
                            annotated=False)

    vectors = read_procssed_anno(vector_file)

    # add the sense embeddings from the vector file to the sense inventory (merging on lobenummer and ddo_dannetsemoid)
    annotations = annotations.merge(vectors[['lobenummer', 'ddo_dannetsemid', 'bert', 'word2vec']],
                                    how='outer',
                                    on=['lobenummer', 'ddo_dannetsemid'])
    # remove senses with missing embeddings
    annotations = annotations[~annotations['bert'].isna()]

    # The embeddings are stacked and then the dimensions are reduced using the reducer
    reducers = {'bert': reducer(np.vstack([np.array(eval(vector)) for i, vector in annotations.bert.items()])),
                'word2vec': reducer(
                    np.vstack([np.array(get_vector_from_string(vector)) for i, vector in annotations.word2vec.items()]))
                }

    return EmbeddingsForVisualisation(annotations, reducers)


class EmbeddingsForVisualisation(object):
    """
    A class used to store and retrieve sense embeddings for visualisation

    Attributes
    ----------
    :attr data:  Pandas groupby object with sense inventory
    :attr reducers: Reducer for dimensionality reduction

    Methods
    -------
    get_representattion(row, vector, max_sense)
        :returns dict with sense representation for visualisation
    get_representation_for_lemmas(self, lemmas, model_name)
        :returns pd.Dataframe with sense represensations for lemmas using model_name
    get_2d_representation_from_lemmas(lemmas, model_name)
        :returns (2dim embeddings, sense levels, sense density, and sense id) for lemmas using model_name
    """

    def __init__(self, data, reducers):
        self.data = data.groupby(['ddo_lemma', 'ddo_homnr'])
        self.reducers = reducers

    @staticmethod
    def get_representation(row, vector, max_sense) -> dict:
        """
         Get a dict with sense embedding and additional information necessary for visualisation

        :param row: row from pandas Dataframe with information for a single sense
        :param vector: (numpy.array) sense embedding
        :param max_sense: (int) counter value (counts how many COR senses already visualised)
        :return: dict"""

        # vector needs to be numpy array
        if isinstance(vector, str):
            if ',' in vector:
                vector = np.array(eval(vector))
            else:
                vector = re.sub('\[ +', '[', vector)
                vector = re.sub('[ \n]+', ', ', vector)
                vector = np.array(eval(vector))

        return {'sense': row.ddo_betyd_nr,  # sense in the Danish Dictionary (DDO)
                'cor': int(row.cor_bet_inventar) + int(max_sense),  # ensures unique colour for each COR sense
                'embedding': vector,
                'length': len(row.bow),
                'lemma': row.ddo_lemma,
                'genprox': row.ddo_genprox,
                'score': row.score,
                'definition': row.ddo_definition}

    def get_representation_for_lemmas(self, lemmas, model_name) -> pd.DataFrame:
        """
        Get a Pandas Dataframe where each row corresponds to a lemma in lemmas and the columns are the information
        necessary for the visualisation.

        :param lemmas: list of (lemma, homonym number)
        :param model_name: embedding model used for creating the embeddings
        :return: pd.DataFrame with sense information for lemmas
        """

        data = []
        extra_lemmas = []  # we add representations for extra lemmas (from genprox + related_words)
        add_n = 0  # how many COR senses that are already represented (ensures different colours for each sense)

        # for every lemma in lemmas
        for index, (lemma, homnr) in enumerate(lemmas):
            group = self.data.get_group((lemma, homnr))
            add_n += int(group.cor_bet_inventar.max())

            # every sense of lemma
            for row in group.itertuples():
                vector = row[group.columns.get_loc(model_name) + 1]  # get the vector calculated by model_name
                content = self.get_representation(row, vector, add_n)  # content == relevant information from DDO
                content['index'] = index + 1
                data.append(content)

                # find the extra lemmas to data
                extra_lemmas.append(row.ddo_genprox)
                stikord = row.cor_stikord.split(',')
                if len(stikord):
                    extra_lemmas += stikord

            extra_lemmas = list(set(extra_lemmas))

        # add the extra lemmas to data
        for index, lemma in enumerate(extra_lemmas):
            for i in range(1, 5):
                if (lemma, i) in self.data.groups:
                    group = self.data.get_group((lemma, i))
                    for row in group.itertuples():
                        vector = row[group.columns.get_loc(model_name) + 1]
                        content = self.get_representation(row, vector, 0)
                        content['cor'] = add_n + 2  # same colour for this group
                        content['score'] = 1  # same size for this group
                        data.append(content)

        return pd.DataFrame(data)

    def get_2d_representations_from_lemmas(self, lemmas, model_name) -> (np.array, list, list, list):
        """
        Get the reduced embeddings, COR sense, dictionary information, score, and index for each lemma in lemmas.

        :param lemmas: list of (lemma, homonym number)
        :param model_name: embedding model used for creating the embeddings
        :return: tuple with embeddings, COR sense, labels, scores, and indices.
        """
        dataset: pd.DataFrame = self.get_representation_for_lemmas(lemmas, model_name)

        dataset = dataset.dropna(subset=['embedding'])  # remove data points with no embeddings
        # reduce the dimensions with the reducer
        senses_2dim = self.reducers[model_name].transform(np.vstack([a for i, a in dataset.embedding.items()]))

        # label to be shown in the visualisation when you hold the mouse over a data point
        labels = [(f"DDO_sense:{row.sense}<br>" +
                   f"COR: {row.cor}<br>" +
                   f"Lemma: {row.lemma}<br>" +
                   f"GenProx: {row.genprox}<br>"
                   f"Definition: {row.definition}<br>" +
                   f"n_words {row.length}<br>')") for row in dataset.itertuples()]

        return senses_2dim, list(dataset['cor']), labels, list(dataset['score']), list(dataset['index'])

    def get_representations_from_lemma(self, lemma: tuple, model_name: str, return_embeddings=True):
        # find lemma in data
        group = self.data.get_group(lemma)

        embeddings = []  # for returning just embeddings
        labels = []  # sense labels when returning just embeddings
        data = []  # for returning all sense information

        for index, row in enumerate(group.itertuples()):
            vector = row[group.columns.get_loc(model_name) + 1]  # get the vector calculated by model_name
            content = self.get_representation(row, vector, index + 1)  # content == relevant information from DDO
            content['index'] = index + 1

            vector = content['embedding']

            if type(vector) != float:  # we only want senses where we have embeddings
                data.append(content)
                embeddings.append(vector)
                labels.append(row.ddo_betyd_nr)

        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        inds = labels.argsort()  # we want the senses in the right order

        if return_embeddings:  # only returns embeddings
            return embeddings[inds], labels[inds]

        else:  # returns all sense information
            return pd.DataFrame(data)

    def get_cosine_matrix(self, lemma, model_name, reduce):
        list_of_senses, labels = self.get_representations_from_lemma(lemma, model_name)
        if reduce:
            list_of_senses = self.reducers[model_name].transform(list_of_senses)
        matrix = pairwise_distances(list_of_senses, metric="cosine")
        return pd.DataFrame(matrix, columns=labels, index=labels)
