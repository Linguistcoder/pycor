import re

import nltk

# nltk.download('stopwords')

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


def clean_ontology(ontology: str) -> set:
    if type(ontology) != str:
        return set()
    return set(ontology.lower().strip('()').split('+'))


def clean_frame(frame: str) -> set:
    if type(frame) != str:
        return set()
    return set(frame.lower().split(';'))


def clean_ddo_bet(ddo_bet: str) -> int:
    if type(ddo_bet) is float:
        return 0
    if any(char.isdigit() for char in ddo_bet):
        sense = re.sub('[^0-9]', '', ddo_bet)
        if sense == '':
            return 0
        return int(sense)
    else:
        return 0

def get_fig_value(fig1, fig2):
    if fig1 == 1:
        if fig2 == 1:
            return 2
        else:
            return 1
    elif fig2 == 1:
        return 1
    else:
        return 0


def get_main_sense(sense):
    if any(char.isdigit() for char in sense) and 'word2vec' not in sense:
        sense = re.sub('[^0-9]', '', sense)
        if sense == '':
            return 0
        return int(sense) * 2
    else:
        return 0


def clean_wcl(wcl):
    if 'sb' in wcl or 'sub' in wcl:
        return 'sb.'
    elif 'vb' in wcl or 'ver' in wcl:
        return 'vb.'
    elif 'adj' in wcl:
        return 'adj.'
