import re
import nltk
nltk.download('stopwords')

stopwords = set(nltk.corpus.stopwords.words('danish'))

def remove_special_char(string: str):
    string = string.lower()
    string = re.sub('el\.', 'eller ', string)
    string = re.sub('lign\.', 'lignende ', string)
    string = re.sub('[()"\'/!?.,;:\-_]+', '', string)
    return string

def remove_stopwords(data, return_str=False):
    if type(data) == str:
        data = data.lower().split()
    data = [token for token in data if token not in stopwords and token.lower() != '[tgt]']
    if return_str is True:
        return ' '.join(data)
    else:
        return list(data)