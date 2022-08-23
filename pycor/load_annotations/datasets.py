import pandas as pd
from collections import namedtuple
from typing import List
import torch
from scipy.spatial.distance import cosine
from pycor.utils import preprocess


class DataSet(List):
    def __init__(self, data, dataset_type, **kwargs):
        super().__init__()
        self.dataset_type = dataset_type

        self.max_sense_limit = kwargs.get('max_sense', 0)
        self.wordclass = kwargs.get('wordclass', 'all')

        if dataset_type == 'sense_selection':
            sents = kwargs.get('sentence_type', 'all')
            self.generate_sense_selection_dataset(data, sents)

        elif dataset_type == 'textbased_only':
            self.generate_textbased_only_dataset(data)

        elif dataset_type == 'rulebased_only':
            self.generate_rulebased_only_dataset(data)

        elif dataset_type == 'feature':
            infotypes = kwargs.get('infotypes', [])
            embeddings = kwargs.get('embedding_type', ['word2vec'])
            self.generate_feature_dataset(data, infotypes, embeddings)

        elif dataset_type == 'bert_reduction':
            sents = kwargs.get('sentence_type', 'all')
            self.generate_bert_reduction_dataset(data, sents)

        elif dataset_type == 'generate_embeddings':
            embeddings = kwargs.get('embedding_type', ['word2vec'])
            output_path = kwargs.get('output_path', 'var/')
            self.generate_embeddings(data, embeddings, output_path)

    def generate_grouped_data(self, groupby):
        pass

    def generate_textbased_only_dataset(self, annotations: pd.DataFrame):

        instance = namedtuple('instance', ['lemma', 'ordklasse', 'homnr', 'bet_1', 'bet1_id', 'bet_2', 'bet2_id',
                                           'sentence_1', 'sentence_2', 'label'])

        print(f"Number of lemmas: {annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']).ngroups}")

        if 'citat' in annotations:
            annotations['citat'] = annotations.citat.apply(lambda x:
                                                           x.replace('[TGT]', '') if isinstance(x, str) else '')

        for name, group in annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']):
            lemma = name[0].lower()
            wcl = name[1].lower()
            hom_nr = name[2]

            if 'pl.' in wcl:
                wcl = wcl.replace(' pl.', '')

            if self.wordclass != 'all' and self.wordclass not in wcl:
                continue

            if 0 < self.max_sense_limit < len(list(group.ddo_betyd_nr.values)):
                continue

            senses = list(zip(group.ddo_betyd_nr.values, group.ddo_dannetsemid.values))

            if 'bow' in group.columns:
                ddo_definitions = {row.ddo_betyd_nr: [row.bow] for row in group.itertuples()}

            else:
                ddo_definitions = {preprocess.remove_special_char(row.ddo_definition) for row in group.itertuples()}

            if 'citat' in annotations:
                ddo_citat = {row.ddo_betyd_nr: row.citat.split('||') for row in group.itertuples() if row.citat != ''}

            ddo2cor = {row.ddo_betyd_nr: 1 for row in group.itertuples()}  # row.cor for row in group.itertuples()}

            sense_pairs = set([frozenset((sens, sens2))
                               for index, sens in enumerate(senses[:-1]) for sens2 in senses[index + 1:]])
            sense_pairs = [pair for pair in sense_pairs if len(pair) > 1]

            for sense, sense2 in sense_pairs:
                sentence_1 = ddo_definitions[sense[0]]  # + ddo_citat.get(sense, [])
                sentence_2 = ddo_definitions[sense2[0]]  # + ddo_citat.get(sense2, [])

                if 'citat' in annotations:
                    sentence_1 += ddo_citat.get(sense[0], [])
                    sentence_2 += ddo_citat.get(sense2[0], [])

                sentence_1 = ' '.join(sentence_1)
                sentence_2 = ' '.join(sentence_2)

                self.append(instance(lemma=lemma,
                                     ordklasse=wcl,
                                     homnr=hom_nr,
                                     bet_1=sense2[0],
                                     bet1_id=sense2[1],
                                     bet_2=sense[0],
                                     bet2_id=sense[1],
                                     sentence_1=sentence_1,
                                     sentence_2=sentence_2,
                                     label=1 if ddo2cor[sense[0]] == ddo2cor[sense2[0]] else 0)
                            )

        return self

    def generate_rulebased_only_dataset(self, annotations: pd.DataFrame):
        """Create a paired word sense dataset based on COR principles.
        The overlap / similarity of each word sense pair is estimated using three rules / principles.
        Two word senses should be merged if:
         1. similar ontological type
         AND
         2. same main sense
         AND
         3. None are figurative senses

        Each output row contains a
        - [0] target lemma,
        - [1] word class,
        - [2] homonym number
        - [3] dictionary sense 1
        - [4] dictionary sense 2
        - [5] ontological type overlap score
        - [6] main sense overlap score
        - [7] figurative score
        - [8] total overlap score
        - [9] label

        :param annotations: pd.DataFrame with columns: ['lemma', 'ordklasse', homnr', 'definition', 'genprox',
                                                        'kollokation', 'cor', 'dn_id', 'ddo_nr', 'citat',
                                                        't_score', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame']
        :return: List of datapoints / training instances
        """

        Sense_pair = namedtuple('Sense_pair',
                                ['lemma', 'ordklasse', 'homnr', 'bet_1', 'bet1_id', 'bet_2', 'bet2_id',
                                 'onto', 'main_sense', 'figurative', 'score', 'label']
                                )

        Sense = namedtuple('Sense', ['lemma', 'ordklasse', 'homnr', 'ddo_bet', 'cor',
                                     'onto', 'main_sense', 'figurative', 'ddo_dannetsemid']
                           )

        for name, group in annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']):

            lemma, wcl, homnr = name
            groupset = []

            if self.wordclass != 'all' and self.wordclass not in wcl:
                continue

            if 'pl.' in wcl:
                wcl = wcl.replace(' pl.', '')

            if 0 < self.max_sense_limit <= len(group.index):
                continue

            for row in group.itertuples():
                figurative = row.ddo_bemaerk if type(row.ddo_bemaerk) != float else ''
                word_sense = Sense(lemma=lemma,
                                   ordklasse=wcl,
                                   homnr=homnr,
                                   ddo_bet=row.ddo_betyd_nr,
                                   cor="",  # row.cor,
                                   onto=preprocess.clean_ontology(row.dn_onto1).union(
                                       preprocess.clean_ontology(row.dn_onto2)),
                                   main_sense=preprocess.get_main_sense(row.ddo_betyd_nr),
                                   figurative=1 if 'ofø' in figurative else 0,
                                   ddo_dannetsemid=row.ddo_dannetsemid
                                   )

                groupset.append(word_sense)

            for indx, sam1 in enumerate(groupset):
                for sam2 in groupset[indx + 1:]:
                    onto_sim = sam1.onto.intersection(sam2.onto)
                    onto_score = 1 if onto_sim == sam1.onto or onto_sim == sam2.onto else 0
                    same_main = 1 if sam1.main_sense == sam2.main_sense else 0
                    both_fig = preprocess.get_fig_value(sam1.figurative, sam2.figurative)
                    label = ""  # 1 if sam1.cor == sam2.cor else 0
                    score = 1 if same_main == 1 and both_fig == 0 and onto_score == 1 else 0

                    pair = Sense_pair(lemma=lemma,
                                      ordklasse=wcl,
                                      homnr=homnr,
                                      bet_1=sam1.ddo_bet,
                                      bet1_id=sam1.ddo_dannetsemid,
                                      bet_2=sam2.ddo_bet,
                                      bet2_id=sam2.ddo_dannetsemid,
                                      onto=onto_score,
                                      main_sense=same_main,
                                      figurative=both_fig,
                                      score=score,
                                      label=label)
                    self.append(pair)

        return self

    def generate_feature_dataset(self, annotations: pd.DataFrame, infotypes, embedding_type):
        """Create a paired word sense feature vector dataset
        Each training instance contains a
            - [0] target lemma,
            - [1] word class,
            - [2] homonym number
            - [3] dictionary sense 1
            - [4] sense 1 id
            - [5] dictionary sense 2
            - [6] sense 2 id
            - [x-z] selected features (information types)
            - [-1] label

        :param annotations: pd.DataFrame with columns: ['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'ddo_definition',
                                                        'cor', ddo_betyd_nr', 'citat', 'onto1', 'onto2', 'hyper',
                                                        'frame', 'ddo_bemaerk']
        :param infotypes:
        :param embedding_type:
        :return: List of datapoints / training instances
        """

        if not isinstance(infotypes, list):
            raise TypeError

        Sense_pair = namedtuple('Sense_pair',
                                ['lemma', 'ordklasse', 'homnr', 'bet_1', 'bet1_id', 'bet_2', 'bet2_id']
                                + infotypes
                                + ['label']
                                )

        Sense = namedtuple('Sense', ['lemma', 'ordklasse', 'homnr', 'ddo_bet', 'bet_id', 'cor', 'word2vec', 'bert',
                                     'onto', 'main_sense', 'figurative']
                           )
        annotations['cor_onto'] = annotations.apply(lambda r: str(r.onto1) + '+' + str(r.onto2), axis=1)

        for name, group in annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']):

            lemma, wcl, homnr = name
            groupset = []

            if 0 < self.max_sense_limit <= len(group.index):
                continue

            if 'pl.' in wcl:
                wcl = wcl.replace(' pl.', '')

            if self.wordclass != 'all' and self.wordclass not in wcl:
                continue

            for row in group.itertuples():
                figurative = row.bemaerk if type(row.bemaerk) != float else ''
                onto = preprocess.clean_ontology(row.onto1).union(preprocess.clean_ontology(row.onto2))
                groupset.append(Sense(lemma=lemma,
                                      ordklasse=wcl,
                                      homnr=homnr,
                                      cor=row.cor,
                                      ddo_bet=row.ddo_betyd_nr,
                                      bet_id=row.sense_id,
                                      word2vec=embedding_type['word2vec']
                                      .embed(embedding_type['word2vec'].tokenizer(row.bow)),
                                      bert=[sent for sent in ('[TGT] ' + row.ddo_definition, row.citat)
                                            if type(row.citat) is not float],
                                      onto=onto,
                                      main_sense=preprocess.get_main_sense(row.ddo_betyd_nr),
                                      figurative=1 if 'ofø' in figurative else 0))

            # pair senses and their information:
            for indx, sam1 in enumerate(groupset):
                for sam2 in groupset[indx + 1:]:
                    onto_sim = sam1.onto.intersection(sam2.onto)
                    onto_score = 1 if onto_sim == sam1.onto else 0.5 if onto_sim == sam2.onto else 0

                    point = [lemma, wcl, homnr, sam1.ddo_bet, sam1.bet_id, sam2.ddo_bet, sam2.bet_id]

                    if 'cosine' in infotypes:
                        point.append(cosine(sam1.word2vec, sam2.word2vec))
                    if 'bert' in infotypes:
                        pairs = [[sam1.lemma, sen1, sen2, 0] for sen1 in sam1.bert for sen2 in sam2.bert]
                        pairs = pd.DataFrame(pairs, columns=['lemma', 'sentence_1', 'sentence_2', 'label'])
                        score = embedding_type['bert'].get_BERT_score(pairs, 'score')
                        point.append(float(torch.mean(score)))
                    if 'onto' in infotypes:
                        point.append(onto_score)
                    if 'main_sense' in infotypes:
                        point.append(1 if sam1.main_sense == sam2.main_sense else 0)
                    if 'figurative' in infotypes:
                        point.append(preprocess.get_fig_value(sam1.figurative, sam2.figurative))

                    point.append(1 if sam1.cor == sam2.cor else 0)

                    self.append(Sense_pair._make(point))
        return self

    def generate_sense_selection_dataset(self, annotations: pd.DataFrame, sentence_type):
        """Creates a Sense Selection dataset from a Dataframe with annotations.

        Each training instance contains a
            - [0] target,
            - [1] a target sentence,
            - [2] a sense quote for each sense in the inventory (list), [3] the sense name in the inventory (list),
            - [4] and the index of the correct sense in the list of senses

        :param annotations: pd.DataFrame with columns: ['lemma', 'ordklasse', homnr', 'definition', 'cor', 'citat',
                                                        'konbet', 'encykl']
        :param sentence_type: whether to include quotes, definitions, or both in examples.
        :return: List of datapoints / training instances
        """
        monosema = 0
        lemma_count = 0
        sense_count1 = 0
        sense_count2 = 0

        noun = 0
        adj = 0
        vb = 0

        lemmas = []

        instance = namedtuple('instance',
                              ['lemma', 'sentence', 'examples', 'senses', 'target', 'wcl'])

        print(f"Number of lemmas: {annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']).ngroups}")

        for name, group in annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']):
            definitions = {}
            examples = {}
            lemma = name[0].lower()
            wcl = name[1].lower()

            if self.wordclass != 'all' and self.wordclass not in wcl:
                continue

            if 0 < self.max_sense_limit < len(list(group.ddo_nr.values)):
                continue

            if 'pl.' in wcl:
                wcl = wcl.replace(' pl.', '')

            for cor, sense_group in group.groupby('cor'):
                definitions[f'COR_{cor}'] = []
                examples[f'COR_{cor}'] = []

                for row in sense_group.itertuples():
                    definitions[f'COR_{row.cor}'] += [f'[TGT] {lemma} [TGT] '
                                                      + preprocess.remove_special_char(row.ddo_definition)]

                    if 'citat' in annotations:
                        if type(row.citat) != float and row.citat != '[]':
                            citats = [row.citat] if '||' not in row.citat else row.citat.split('||')
                            examples[f'COR_{row.cor}'] += citats
                        else:
                            print(f'Missing citat for {lemma}: {row.ddo_definition}')

            # todo: optimize
            if sentence_type == 'all' or sentence_type == 'def':
                sense_count1 += len(definitions)
                if not len(definitions) > 1:
                    monosema += 1
                    continue
                sense_count2 += len(definitions)

                for index, (s, defi) in enumerate(definitions.items()):
                    if len(defi) > 1:
                        for extra_def in defi[1:]:
                            self.append(instance(lemma=lemma,
                                                 sentence=extra_def,
                                                 examples="||".join([d[0] for d in definitions.values()]),
                                                 senses="||".join(definitions),
                                                 target=index,
                                                 wcl=wcl
                                                 ))

            if sentence_type == 'exam':
                sense_count1 += sum([1 for exam, examlist in examples.items() if len(examlist) > 0])
                if not len(definitions) > 1:
                    monosema += 1
                    continue

            if sentence_type == 'exam' or sentence_type == 'all':
                for index, (s, example_list) in enumerate(examples.items()):
                    if sentence_type == 'exam':
                        sense_count2 += 1 if len(example_list) > 0 else 0
                    for example in example_list:
                        self.append(instance(lemma=lemma,
                                             sentence=example,
                                             examples="||".join([de[0] for de in definitions.values()]),
                                             senses="||".join(definitions),
                                             target=index,
                                             wcl=wcl))

            noun += 1 if 'sb.' in wcl else 0
            adj += 1 if 'adj.' in wcl else 0
            vb += 1 if 'vb.' in wcl else 0

            lemma_count += 1
            lemmas.append((lemma, wcl))

        print(f'Number of monosemic lemmas: {monosema}')
        print(f'Number of lemmas: {lemma_count}')
        print(f'Number of senses: {sense_count1}')
        print(f'Number of poly senses: {sense_count2}')
        print(f'Number of nouns: {noun}')
        print(f'Number of verbs: {vb}')
        print(f'Number of adjectives: {adj}')

        return self

    def generate_bert_reduction_dataset(self, annotations, sentence_type):

        instance = namedtuple('instance', ['lemma', 'ordklasse', 'homnr', 'bet_1', 'bet1_id', 'bet_2', 'bet2_id',
                                           'sentence_1', 'sentence_2', 'label'])

        print(f"Number of lemmas: {annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']).ngroups}")

        for name, group in annotations.groupby(['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr']):
            lemma = name[0].lower()
            wcl = name[1].lower()
            hom_nr = name[2]

            if self.wordclass != 'all' and self.wordclass not in wcl:
                continue

            if 0 < self.max_sense_limit < len(list(group.ddo_betyd_nr.values)):
                continue

            if 'pl.' in wcl:
                wcl = wcl.replace(' pl.', '')

            senses = list(zip(group.ddo_betyd_nr.values, group.ddo_dannetsemid.values))

            ddo_definitions = {row.ddo_betyd_nr: [f'[TGT] {lemma} [TGT] '
                                                  + preprocess.remove_special_char(row.ddo_definition)]
                               for row in group.itertuples()}

            if 'citat' in annotations:
                ddo_citat = {row.ddo_nr: row.citat.split('||') for row in group.itertuples()
                             if type(row.citat) == str and row.citat != '[]'}
            else:
                ddo_citat = {}

            # if label not available
            ddo2cor = {row.ddo_betyd_nr: 1 for row in group.itertuples()}  # row.cor for row in group.itertuples()}

            sense_pairs = set([frozenset((sens, sens2))
                               for index, sens in enumerate(senses[:-1]) for sens2 in senses[index + 1:]])
            sense_pairs = [pair for pair in sense_pairs if len(pair) > 1]

            for sense, sense2 in sense_pairs:
                sentences = []
                if sentence_type == 'def' or sentence_type == 'all':
                    sentences += ddo_definitions[sense2[0]]
                if sentence_type == 'exam' or sentence_type == 'all':
                    sentences += ddo_citat.get(sense2[0], [])

                for sentence in sentences:
                    if sentence:
                        self.append(instance(lemma=lemma,
                                             ordklasse=wcl,
                                             homnr=hom_nr,
                                             bet_1=sense2[0],
                                             bet1_id=sense2[1],
                                             bet_2=sense[0],
                                             bet2_id=sense[1],
                                             sentence_1=sentence,
                                             sentence_2=ddo_definitions[sense[0]][0],
                                             label=1 if ddo2cor[sense[0]] == ddo2cor[sense2[0]] else 0))

        return self

    def generate_embeddings(self, annotations, embedding_type, output_path):

        if 'bert' in embedding_type:
            print('Calculating BERT embeddings')
            annotations['bert'] = annotations.apply(lambda row: embedding_type['bert'].get_bert_embedding(row),
                                                    axis=1)
            print('Added BERT embeddings')
        if 'word2vec' in embedding_type:
            print('Calculating word2vec embeddings')
            annotations['word2vec'] = annotations.apply(lambda row: embedding_type['word2vec']
                                                        .embed(embedding_type['word2vec']
                                                               .tokenizer(row.bow)),
                                                        axis=1)
            print('Added word2vec embeddings')
        annotations.to_csv(f'{output_path}/annotations_with_embeddings.tsv', sep='\t', encoding='utf8')

        return annotations

    def to_dataframe(self):
        return pd.DataFrame(self)

    def to_tsv(self, filename):
        return self.to_dataframe().to_csv(filename, sep='\t', encoding='utf8')
