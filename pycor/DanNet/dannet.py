from typing import OrderedDict

from pycor.DanNet.dan_utils import remove_end
from pycor.utils.preprocess import remove_stopwords, remove_special_char


class Synset(object):
    def __init__(self, synset_id, wordforms, word_ids, gloss, ont, synonyms=[],
                 hyponyms=[], hypernyms=[], holonyms=[], meronyms=[], other_relations=[]):
        self.synset_id = synset_id
        self.wordforms = wordforms
        self.word_ids = word_ids
        self.synonyms = synonyms
        self.hyponyms = hyponyms
        self.hypernyms = hypernyms
        self.holonyms = holonyms
        self.meronyms = meronyms
        self.gloss = gloss
        self.ont = ont

        self.other_rel_dict = {rel[1]: rel[0] for rel in other_relations}
        self.other_relations = list(self.other_rel_dict.keys())

    def __str__(self):
        string = f"""Synset: {self.wordforms[0]}\n
                    n_words: {str(len(self.wordforms))}\n
                    n_relations: {str(len(self.other_relations) + len(self.hyponyms) + len(self.hypernyms) +
                                      len(self.holonyms) + len(self.meronyms))}\n
                    gloss: {self.gloss}
                    """
        return string

    def __repr__(self):
        return f"Synset(forms={','.join(self.wordforms)}, id={self.synset_id})"

    def get_relations(self, relation_type=None):
        if relation_type == 'hypernym' or relation_type == 'hyper':
            return self.hypernyms
        elif relation_type == 'hyponym' or relation_type == 'hypo':
            return self.hyponyms
        elif relation_type == 'hh' or relation_type == 'hyper-hypo':
            return self.hypernyms + self.hyponyms
        elif relation_type == 'holonyms' or relation_type == 'holo':
            return self.holonyms
        elif relation_type == 'meronyms' or relation_type == 'mero':
            return self.meronyms
        elif relation_type == 'composition':
            return self.holonyms + self.meronyms
        elif relation_type == 'other':
            return self.other_relations
        else:
            return self.hypernyms + self.hyponyms + self.holonyms + self.meronyms + self.other_relations

    def get_definition(self, preprocess=True):
        if type(self.gloss) == float:
            return []
        elif self.gloss == '(ingen definition)':
            return []
        elif 'Brug:' in self.gloss:
            definition = self.gloss.split('(Brug:')[0]
            definition = remove_end(definition)
            if preprocess is False:
                return [definition]
            else:
                return remove_stopwords(remove_special_char(definition))
        else:
            definition = remove_end(self.gloss)
            if preprocess is False:
                return [definition]
            else:
                return remove_stopwords(remove_special_char(definition))

    def get_example_sentence(self, preprocess=True):
        if type(self.gloss) == float:
            return []
        elif self.gloss == '(ingen definition)':
            return []
        elif 'Brug:' in self.gloss:
            sentences = self.gloss.split('(Brug:')[1]
            sentences = sentences.replace(';', '||')
            if '||' in sentences:
                sentences = sentences.split('||')
                if preprocess is False:
                    return [s.replace('"', '') for s in sentences]
                sentences = [remove_stopwords(remove_special_char(s)) for s in sentences]
                return [token for sent in sentences for token in sent]
            else:
                if preprocess is False:
                    return [sentences.replace('"', '')]
                else:
                    return remove_stopwords(remove_special_char(sentences))
        else:
            return self.get_definition(preprocess=preprocess)

    def get_all_example_sentences(self, hypernyms=False, hyponyms=False):
        all_examples = []
        all_examples += self.get_example_sentence(preprocess=False)

        if hypernyms:
            for hypernym in self.hypernyms:
                all_examples += hypernym.get_example_sentence(preprocess=False)

        if hyponyms:
            for hyponym in self.hyponyms:
                all_examples += hyponym.get_example_sentence(preprocess=False)

        return all_examples

    def get_members(self):
        members = []
        members += self.wordforms
        for hyp in self.hypernyms + self.hyponyms:
            members += hyp.wordforms
        for rel in self.other_relations + self.synonyms + self.holonyms + self.meronyms:
            members += rel.wordforms
        return members


class DanNet(OrderedDict):
    def __init__(self):
        super().__init__()

    def get_top_node(self):
        return self[20633]

    def get_top_node_childs(self):
        top_node = self.get_top_node()
        return top_node.hyponyms
