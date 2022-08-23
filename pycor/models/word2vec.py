from abc import ABC

import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine

from pycor.utils.preprocess import remove_special_char, remove_stopwords


class word2vec_model(KeyedVectors):
    def tokenizer(self, sentence):
        output = [word for word in remove_stopwords(remove_special_char(sentence))
                  if word in self.key_to_index]
        return output

    def embed(self, sentence):
        if type(sentence) == str:
            sentence = [sentence]
        elif type(sentence) == float:
            return np.nan
        vectors = [self.get_vector(word) for word in sentence if word in self.key_to_index]
        if len(vectors) < 1:
            return np.nan
        return np.mean(vectors, axis=0)

    def vectorize_and_cosine(self, sentence_1, sentence_2):
        vector1 = self.embed(self.tokenizer(sentence_1))
        vector2 = self.embed(self.tokenizer(sentence_2))

        return cosine(vector1, vector2)

# datasets = ['cbc_train', 'cbc_devel', 'cbc_test', 'keywords_train', 'keywords_test',
#             'mellem_train', 'mellem_test', 'cbc_devel_less', 'cbc_test_less', 'keywords_test_less'
#             ]

# for subset in datasets:
#      print(f'______________________{subset.upper()}____________________________')
#      train_dataset = pd.read_csv(f'../../data/reduction/reduction_word2vec_{subset}.tsv', '\t', encoding='utf8')
#      train_dataset['score'] = train_dataset.apply(lambda row: vectorize_and_cosine(row), axis=1)
#      train_dataset.to_csv(f'../../data/word2vec/reduction_score_{subset}.tsv', sep='\t')
