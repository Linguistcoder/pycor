import requests
import re
import spacy

nlp = spacy.load("da_core_news_sm")

URL = 'https://cst.dk/tools/index.php'

POST = {'MAX_FILE_SIZE': 8000000,
        'inputform': 'j',
        'language': 'da',
        'password': '',
        'inputText': '',
        'inputFile': '(binary)',
        'token': 'j',
        'navne': 'j',
        'pos': 'j',
        'lemma': 'j',
        'anonym': 'n',
        'abbr': 'j',
        'mwu': 'j',
        'what': 'b',
        'sorting': 'nosort',
        'ambi': 'n',
        'dict': 'j'
        }


def get_lemmatised_sentence(sentence, post=POST, url=URL):
    post['inputText'] = sentence
    req = requests.post(url, data=post)

    result = re.findall('"auto">.+\n', req.text)[0][7:]

    return result


def form_in_sentence(sentence, form):
    """Checks that a word form is in a sentence"""

    sentence = sentence.strip()
    lan = nlp(sentence)
    lemma_sentence = [token.lemma_ for token in lan]
    text_sentence = [token.text for token in lan]
    form = nlp(form)[0].lemma_

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
            form_index = [i for i, w in enumerate(lemma_sentence) if w == form]
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

# def form_in_sentence2(sentence, form):
#     """Checks that a word form is in a sentence"""
#
#     sentence = sentence.strip()
#     lemma_sentence = get_lemmatised_sentence(sentence).split()
#
#     if form in lemma_sentence:
#         form_index = lemma_sentence.index(form)
#         text_sentence = [s if i != form_index else f'[TGT] {s} [TGT]' for i, s in enumerate(sentence.split())]
#         return ' '.join(text_sentence)
#
#     else:
#         print(f'Form "{form}" cannot be found in the sentence:\n{sentence}')
#         #sentence_input = input('Please input correct sentence:')
#         sentence_input = 'no target'
#         return sentence_input
