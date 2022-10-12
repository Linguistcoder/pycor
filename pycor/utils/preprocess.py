import re
import nltk

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('danish'))

def remove_special_char(string: str):
    """removes special characters from string and expands common abbreviations"""
    if type(string) != str:
        string = str(string)
    string = string.lower()
    string = re.sub(r'el\.', 'eller ', string)
    string = re.sub('lign\.', 'lignende ', string)
    string = re.sub('mht', 'med hensyn til ', string)
    string = re.sub('[/]', ' ', string)
    string = re.sub('[()"\'!?.,;:_\[\]\-]+', '', string)
    string = re.sub('  +', ' ', string)
    return string


def remove_stopwords(data, return_str=False):
    """
    removes stopwords from data. Stopword defined by the NLTK-package
    :param data: (str or list) text data that needs stopword removal
    :param return_str: (bool) whether to return as string (True) or list (False)
    """
    if type(data) == str:
        data = data.lower().split()
    data = [token for token in data if token not in stopwords and token.lower() != '[tgt]']  # ignore [tgt]
    if return_str is True:
        return ' '.join(data)
    else:
        return list(data)


def clean_ontology(ontology: str) -> set:
    """cleans ontological type by lower casing, remove parentheses, and +
    :returns: ontological type as a set"""
    if type(ontology) != str:
        return set()
    return set(ontology.lower().strip('()').split('+'))


def get_fig_value(fig1, fig2):
    """returns whether fig1 or fig2 is figurative"""
    if fig1 == 1 or fig2 == 1:
        return 1
    else:
        return 0


def get_main_sense(sense):
    """returns the main sense (int) of sense (removes .a .b etc.)"""
    if any(char.isdigit() for char in sense) and 'word2vec' not in sense:
        sense = re.sub('[^0-9]', '', sense)
        if sense == '':
            return 0
        return int(sense) * 1
    else:
        return 0

