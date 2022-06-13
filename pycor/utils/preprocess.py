import re
import spacy
import nltk

from pycor.utils.lemmatizer import get_lemmatised_sentence

nlp = spacy.load("da_core_news_sm")
# nltk.download('stopwords')

stopwords = set(nltk.corpus.stopwords.words('danish'))


def remove_special_char(string: str):
    if type(string) != str:
        string = str(string)
    string = string.lower()
    string = re.sub('el\.', 'eller ', string)
    string = re.sub('lign\.', 'lignende ', string)
    string = re.sub('mht', 'med hensyn til ', string)
    string = re.sub('[/]', ' ', string)
    string = re.sub('[()"\'!?.,;:_\[\]\-]+', '', string)
    string = re.sub('  +', ' ', string)
    return string


def clean_str(string):
    if type(string) != str:
        string = str(string)
    string = string.lower()
    string = re.sub('el\.', 'eller ', string)
    string = re.sub('lign\.', 'lignende ', string)
    string = re.sub('mht', 'med hensyn til ', string)
    string = re.sub('[/]', ' ', string)
    string = re.sub('[()"\'!?.,;:_\[\]\-]+', '', string)
    string = re.sub('\s+', ' ', string)
    list_string = remove_stopwords(string)
    return '||'.join(list_string)


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
        #if fig2 == 1:
        #    return 2
        #else:
        #   return 1
        return 1
    elif fig2 == 1:
        return 1
    else:
        return 0


def get_main_sense(sense):
    try:
        if any(char.isdigit() for char in sense) and 'word2vec' not in sense:
            sense = re.sub('[^0-9]', '', sense)
            if sense == '':
                return 0
            return int(sense) * 2
    except:
        print(sense)
    else:
        return 0


def clean_wcl(wcl):
    if 'sb' in wcl or 'sub' in wcl:
        return 'sb.'
    elif 'vb' in wcl or 'ver' in wcl:
        return 'vb.'
    elif 'adj' in wcl:
        return 'adj.'

def form_in_sentence(sentence, form):
    """Checks that a word form is in a sentence"""

    sentence = sentence.strip()
    lan = nlp(sentence)
    lemma_sentence = [token.lemma_ for token in lan]
    text_sentence = [token.text for token in lan]
    #form = nlp(form)[0].lemma_

    if form in lemma_sentence:
        form_index = [i for i, w in enumerate(lemma_sentence) if w == form]
        text_sentence = [s if i not in form_index else f'[TGT] {s} [TGT]' for i, s in enumerate(text_sentence)]
        return ' '.join(text_sentence)

    elif form in text_sentence:
        form_index = [i for i, w in enumerate(text_sentence) if w == form]
        text_sentence = [s if i not in form_index else f'[TGT] {s} [TGT]' for i, s in enumerate(text_sentence)]
        return ' '.join(text_sentence)

    else:
        lemma_sentence = get_lemmatised_sentence(sentence).split()
        if form in lemma_sentence:
            form_index = [i for i, w in enumerate(lemma_sentence) if w==form]
            text_sentence = [s if i not in form_index else f'[TGT] {s} [TGT]' for i, s in enumerate(sentence.split())]
            return ' '.join(text_sentence)

        else:
            print(f'Form "{form}" cannot be found in the sentence:\n{sentence}')
            print([(i, s) for i, s in enumerate(sentence.split())])

            form_index = input('Please input correct index: ')
            form_index = int(form_index) if form_index != '' else 0
            text_sentence = [s if i != form_index else f'[TGT] {s} [TGT]' for i, s in enumerate(sentence.split())]
            print(' '.join(text_sentence) + '\n')
            return ' '.join(text_sentence)



def form_in_sentence2(sentence, form):
    """Checks that a word form is in a sentence"""

    sentence = sentence.strip()
    lemma_sentence = get_lemmatised_sentence(sentence).split()

    if form in lemma_sentence:
        form_index = lemma_sentence.index(form)
        text_sentence = [s if i != form_index else f'[TGT] {s} [TGT]' for i, s in enumerate(sentence.split())]
        return ' '.join(text_sentence)

    else:
        print(f'Form "{form}" cannot be found in the sentence:\n{sentence}')
        #sentence_input = input('Please input correct sentence:')
        sentence_input = 'no target'
        return sentence_input
