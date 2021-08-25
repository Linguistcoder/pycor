from gensim.models import KeyedVectors
import numpy as np


#model_path = 'data/word2vec/dsl_skipgram_2020_m5_f500_epoch2_w5.model'
#word2vec = Word2Vec.load(model_path).wv
from pycor.utils.preprocess import remove_special_char, remove_stopwords

print('Loading word2vec model')
model_path = 'data/word2vec/dsl_skipgram_2020_m5_f500_epoch2_w5.model.txtvectors'
word2vec = KeyedVectors.load_word2vec_format(model_path,
                                             fvocab='data/word2vec/dsl_skipgram_2020_m5_f500_epoch2_w5.model.txtvectors.vocab',
                                             binary=False,
                                             limit=1000000)
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
