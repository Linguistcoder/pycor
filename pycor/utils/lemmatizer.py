import requests
import re

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
