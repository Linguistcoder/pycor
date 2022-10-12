import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine

from pycor.utils.preprocess import remove_special_char, remove_stopwords


class word2vec_model(KeyedVectors):
    """Class for loading and embedding with word2vec model

    Methods
    -------
    tokenizer(self, sentence)
        :returns: tokenized output

    embed(self, sentence)
        :returns: embedded sentence

    vectorize_and_cosine(self, sentence_1, sentence_2)
        :returns: cosine similarity between the embedded sentence_1 and sentence_2
    """
    def tokenizer(self, sentence):
        """Preprocess and tokenize sentence"""
        output = [word for word in remove_stopwords(remove_special_char(sentence))
                  if word in self.key_to_index]
        return output

    def embed(self, sentence):
        """embed the sentence by calculating the centroid (average) embedding of each token in sentence"""
        if type(sentence) == str:  # type has to be list (or similar)
            sentence = [sentence]
        elif type(sentence) == float:  # if no sentence -> return nan
            return np.nan
        # vectorize each token in sentence
        vectors = [self.get_vector(word) for word in sentence if word in self.key_to_index]
        if len(vectors) < 1:
            return np.nan
        # calculate centroid embedding
        return np.mean(vectors, axis=0)

    def vectorize_and_cosine(self, sentence_1, sentence_2):
        """calculate cosine similarity between embedded sentence_1 and embedded sentence_2"""
        vector1 = self.embed(self.tokenizer(sentence_1))
        vector2 = self.embed(self.tokenizer(sentence_2))

        return cosine(vector1, vector2)
