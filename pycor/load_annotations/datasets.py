import random
import pandas as pd
import numpy as np
from collections import namedtuple
from scipy.spatial.distance import cosine
from typing import List

from pycor.utils import preprocess
from pycor.utils.vectors import vectorize


class DataSet(List):
    def __init__(self, data, dataset_type, **kwargs):
        self.dataset_type = dataset_type
        self.instance = ()

        if self.dataset_type == 'sense_selection':
            super(DataSet, self.generate_sense_selection_dataset(data)).__init__()

        elif dataset_type == 'textbased_only':
            super(DataSet, self.generate_textbased_only_dataset(data)).__init__()

        elif dataset_type == 'feature':
            super(DataSet, self.generate_feature_dataset(data, kwargs)).__init__()

    def generate_grouped_data(self, groupby):
        pass

    def generate_textbased_only_dataset(self, annotations: pd.DataFrame):
        pass

    def generate_feature_dataset(self, annotations: pd.DataFrame, kwargs):
        """Create feature vector dataset
        Each training instance contains a
            - [0] target lemma,
            - [1] word class,
            - [2] homonym number
            - [3] dictionary sense 1
            - [4] dictionary sense 2
            - [x-z] selected features (information types)
            - [-1] label

        :param annotations: pd.DataFrame with columns: ['lemma', 'ordklasse', homnr', 'definition', 'genprox',
                                                        'kollokation', 'cor', 'dn_id', 'ddo_nr', 'citat',
                                                        't_score', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame']
        :param kwargs: keyword arguments: infotypes, textbase, max_sense
        :return: List of datapoints / training instances
        """

        infotypes = kwargs['infotypes']
        textbase = kwargs['textbase']

        self.instance = namedtuple('instance',
                                   ['lemma', 'ordklasse', 'homnr', 'bet_1', 'bet_2']
                                   + infotypes
                                   + ['label']
                                   )

        sense = namedtuple('sense', ['lemma', 'ordklasse', 'cor', 'ddo_bet', 'vector',
                                     'onto', 'frame', 'score', 'figurative']
                           )

        for name, group in annotations.groupby(['lemma', 'ordklasse', 'homnr']):
            groupset = []

            if 'max_sense' in kwargs and len(group.index) > int(kwargs['max_sense']):
                continue

            for row in group.itertuples():
                figurative = row.bemaerk if type(row.bemaerk) != float else ''

                groupset.append(sense(lemma=row.lemma,
                                      ordklasse=row.ordklasse,
                                      cor=row.cor,
                                      ddo_bet=row.ddo_nr,
                                      vector=vectorize(row, infotypes=textbase),
                                      onto=preprocess.clean_ontology(row.onto1) \
                                      .union(preprocess.clean_ontology(row.onto2)),
                                      frame=preprocess.clean_frame(row.frame),
                                      score=int(row.t_score),
                                      figurative=1 if 'ofø' in figurative else 0))

            # pair senses and their information:
            for indx, sam1 in enumerate(groupset):
                for sam2 in groupset[indx + 1:]:
                    onto_len = len(sam1.onto.intersection(sam2.onto))
                    frame_len = len(sam1.frame.intersection(sam2.frame))

                    point = [sam1.lemma, sam1.wcl, name[3], sam1.ddo_bet, sam2.ddo_bet]

                    if 'cosine' in infotypes:
                        point.append(cosine(sam1.vector, sam2.vector))
                    if 'onto' in infotypes:
                        point.append(2 if onto_len == len(sam1.onto) else 1 if onto_len >= 1 else 0)
                    if 'frame' in infotypes:
                        point.append(2 if frame_len == len(sam1.frame) else 1 if frame_len >= 1 else 0)
                    if 'main_sense' in infotypes:
                        point.append(1 if sam1.main_sense == sam2.main_sense else 0)
                    if 'figurative' in infotypes:
                        point.append(preprocess.get_fig_value(sam1.figurative, sam2.figurative))

                    point.append(1 if sam1.cor == sam2.cor else 0)
                    self.append(self.instance(point))
        return self

    def generate_sense_selection_dataset(self, annotations: pd.DataFrame):
        """Creates a Sense Selection dataset from a Dataframe with annotations.

        Each training instance contains a
            - [0] target,
            - [1] a target sentence,
            - [2] a sense quote for each sense in the inventory (list), [3] the sense name in the inventory (list),
            - [4] and the index of the correct sense in the list of senses

        :param annotations: pd.DataFrame with columns: ['lemma', 'ordklasse', homnr', 'definition', 'cor', 'citat',
                                                        'konbet', 'encykl']
        :return: List of datapoints / training instances
        """

        self.instance = namedtuple('instance',
                                   ['target', 'target_sentence', 'sense_quotes', 'sense_labels', 'label_index'])

        # todo: check annotations has the right columns
        # we want to treat each lemma as its own group
        for name, group in annotations.groupby(['lemma', 'ordklasse', 'homnr']):
            definitions = {}
            examples = {}
            lemma = name[0].lower()

            # we group together the rows for the lemma that has the same sense
            for cor, sense_group in group.groupby('cor'):
                definitions[f'COR_{cor}'] = []
                examples[f'COR_{cor}'] = []

                for row in sense_group.itertuples():
                    # only add konbet and encykl if they are not empty
                    definition = row.definition
                    # definition might be longer (but data is saved in other columns)
                    if type(row.konbet) == str:
                        definition += row.konbet
                    if type(row.encykl) == str:
                        definition += row.encykl

                    # all definitions and quotes are preprocessed
                    definitions[f'COR_{row.cor}'] += [
                        f'[TGT] {lemma} [TGT] ' + preprocess.remove_special_char(definition)]

                    if type(row.citat) != float:
                        citats = [row.citat] if '||' not in row.citat else row.citat.split('||')
                        examples[f'COR_{row.cor}'] += [
                            preprocess.form_in_sentence(preprocess.remove_special_char(citat),
                                                        lemma) for citat in citats]
            # skip if there is only one sense
            if not len(definitions) > 1:
                continue

            # pair definitions with secondary definitions
            for index, (s, defi) in enumerate(definitions.items()):
                if len(defi) > 1:
                    for extra_def in defi[1:]:
                        self.append(self.instance(target=lemma,
                                                  target_sentence=extra_def,
                                                  sense_quotes="||".join([de[0] for de in definitions.values()]),
                                                  sense_labels="||".join(definitions),
                                                  label_index=index
                                                  ))
            # pair definitions with quotes
            for index, (s, example_list) in enumerate(examples.items()):
                for example in example_list:
                    self.append(self.instance(target=lemma,
                                              target_sentence=example,
                                              sense_quotes="||".join([de[0] for de in definitions.values()]),
                                              sense_labels="||".join(definitions),
                                              label_index=index))
                    # pair quotes with secondary definitions
                    if len(definitions[s]) > 1:
                        for extra_def in definitions[s][1:]:
                            new_defi_list = [
                                de[0] if len(de) == 1 else random.choice(de[1:]) if sens != s else extra_def
                                for sens, de in definitions.items()]

                            self.append(self.instance(target=lemma,
                                                      target_sentence=example,
                                                      sense_quotes="||".join(new_defi_list),
                                                      sense_labels="||".join(definitions),
                                                      label_index=index
                                                      ))

        return self

    def to_dataframe(self):
        return pd.DataFrame(self)

    def to_tsv(self, filename):
        return self.to_dataframe().to_tsv(filename, sep='\t', encoding='utf8')
