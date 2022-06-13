import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

from pycor.config import Config
from pycor.utils.preprocess import remove_special_char, remove_stopwords

#model_path = 'data/word2vec/dsl_skipgram_2020_m5_f500_epoch2_w5.model'
#word2vec = Word2Vec.load(model_path).wv

print('Loading word2vec model')
model_path = Config.word2vec_PATH
word2vec = KeyedVectors.load_word2vec_format(model_path,
                                             fvocab=Config.word2vec_PATH + '.vocab',
                                             binary=False)
print('Loaded word2vec model')

def word2vec_tokenizer(sentence):
    output = [word for word in remove_stopwords(remove_special_char(sentence))
              if word in word2vec.key_to_index]
    return output


def word2vec_embed(sentence):
    if type(sentence) == str:
        sentence = [sentence]
    elif type(sentence) == float:
        return np.nan
    vectors = [word2vec.get_vector(word) for word in sentence if word in word2vec.key_to_index]
    if len(vectors) < 1:
        return np.nan
    return np.mean(vectors, axis=0)


def vectorize_and_cosine(row):
    vector1 = word2vec_embed(word2vec_tokenizer(row.sentence_1))
    vector2 = word2vec_embed(word2vec_tokenizer(row.sentence_2))

    return cosine(vector1, vector2)


datasets = ['cbc_train', 'cbc_devel', 'cbc_test', 'keywords_train', 'keywords_test',
             'mellem_train', 'mellem_test', 'cbc_devel_less', 'cbc_test_less', 'keywords_test_less'
             ]

# for subset in datasets:
#      print(f'______________________{subset.upper()}____________________________')
#      train_dataset = pd.read_csv(f'../../data/reduction/reduction_word2vec_{subset}.tsv', '\t', encoding='utf8')
#      train_dataset['score'] = train_dataset.apply(lambda row: vectorize_and_cosine(row), axis=1)
#      train_dataset.to_csv(f'../../data/word2vec/reduction_score_{subset}.tsv', sep='\t')