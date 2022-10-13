from pathlib import Path

import pandas as pd

from config import Configuration
from pycor.utils.preprocess import remove_special_char
from typing import Optional, List
import random

random.seed(81)


def read_anno(anno_file: Path, quote_file: Path, columns: Optional[List] = None,
              annotated: bool = True) -> pd.DataFrame:
    """
    load and clean anno_file.
    If quote_file is specified, then quote_file is merged with anno_file on ['ddo_dannetsemid'].

    :param anno_file: (str) filename of annotation file
    :param quote_file: (str) filename of files of cleaned quotes
    :param columns: (list) of columns to keep
    :param annotated: (bool) whether the annotation file is annotated (has column cor_bet_inventar)
    :return: pd.DataFrame of anno_file data
    """

    if columns is None:
        columns = []

    anno = pd.read_csv(anno_file,
                       sep='\t',
                       encoding='utf8',
                       na_values=['n', ' '],  # n is deleted lemmas
                       index_col=False
                       )

    if annotated is True:
        # each row has to be annotated. '0' == deleted senses
        anno = anno.dropna(subset=['ddo_lemma', 'cor_bet_inventar', 'ddo_dannetsemid'])
        anno = anno[anno['cor_bet_inventar'] != '0']
    else:
        anno = anno.dropna(subset=['ddo_lemma', 'ddo_dannetsemid'])

    anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')  # same type for when we merge dataframes
    # if no homonyms in the dictionary, then this field is empty. But we want it to have a number, so we fill with 1
    anno['ddo_homnr'] = anno['ddo_homnr'].fillna(1)

    # for bag-of-words representation
    bow = ['ddo_lemma', 'ddo_definition', 'ddo_genprox']

    if quote_file:
        quotes = pd.read_csv(quote_file,
                             sep='\t',
                             encoding='utf-8',
                             usecols=['ddo_dannetsemid', 'citat']
                             )
        # some senses have multiple quotes. We put them in same field but separated by '||'
        quotes = quotes.groupby('ddo_dannetsemid').aggregate('||'.join)
        # add quotes to annotaiton
        anno = anno.merge(quotes, how='outer', on=['ddo_dannetsemid'])

        # quotes without '[TGT]' and as one long sequence
        anno['citat2'] = anno.citat.apply(lambda x: x.replace('[TGT]', '') if isinstance(x, str) else '')
        anno['citat2'] = anno.citat2.apply(lambda x: x.replace('||', '') if isinstance(x, str) else '')
        bow += ['citat2']

    # whether data has been added from the Danish Thesaurus
    if 'ddb_afsnit' in anno and 'ddb_stikord_brutto' in anno:
        # clean thesaurus data (separated by white space and no special characters)
        anno["ddb_stikord"] = anno.ddb_stikord_brutto.apply(
            lambda x: x.replace(',', ' ').lower() if isinstance(x, str) else '')
        anno.ddb_stikord = anno.ddb_stikord.apply(remove_special_char)
        anno.ddb_afsnit = anno.ddb_afsnit.apply(lambda x: x.replace(';', ' ').lower() if isinstance(x, str) else '')
        anno.ddb_afsnit = anno.ddb_afsnit.apply(remove_special_char)
        bow += ['ddb_afsnit', 'ddb_stikord']

    # ensure merging has not added extra empty rows
    anno = anno.dropna(subset=['ddo_lemma', 'ddo_genprox', 'ddo_dannetsemid', 'ddo_betyd_nr'])

    # clean and expand defintions (ddo_konbet + ddo_encykl contains extra info for definitions)
    anno.ddo_konbet = anno.ddo_konbet.fillna('').astype(str)
    anno.ddo_encykl = anno.ddo_encykl.fillna('').astype(str)
    anno.ddo_definition = anno.ddo_definition.fillna('').astype(str)
    anno.ddo_definition = anno[['ddo_definition', 'ddo_konbet', 'ddo_encykl']].aggregate(' '.join, axis=1)
    anno.ddo_definition = anno.ddo_definition.apply(remove_special_char)

    anno['bow'] = anno[bow].aggregate(' '.join, axis=1)
    anno['length'] = anno.bow.apply(lambda x: len([i for i in x.split('||') if i != '']))

    anno = anno.fillna('')

    if len(columns):
        anno = anno.loc[:, columns]

    return anno


def read_procssed_anno(anno_file: str) -> pd.DataFrame:
    """read anno_file that has already been processed by read_anno()"""
    anno = pd.read_csv(anno_file,
                       sep='\t',
                       encoding='utf8',
                       na_values=[' ']
                       )

    return anno


def sample(population, sample_size, bias, **subgroup):
    # group by lemmas
    lemma_groups = population.groupby(['ddo_lemma', 'ddo_homnr'])
    # only use polysemous lemmas
    groups = [idx for idx in lemma_groups.groups.values() if len(idx) > 1]

    # how many lemmas to sample
    n_samples = int(len(groups) * sample_size)
    # bias levels the difference between n_senses in subsamples
    sampler = random.sample(groups, k=n_samples + bias)
    # get indices for first subsample
    indices = [i for idx in sampler for i in idx]

    # get subsamples
    sample1 = population[population.index.isin(indices)]
    sample2 = population[~population.index.isin(indices)]

    if subgroup:  # subgroup splits test (sample2) into two (devel, test)
        subsample_size = subgroup.get('subsample_size', 0)
        sub_bias = subgroup.get('sub_bias', 0)

        if subsample_size > 0:
            subsample1, subsample2 = sample(sample2, subsample_size, sub_bias)

            print('train_length:', len(sample1))
            print('devel_length:', len(subsample1))
            print('test_length:', len(subsample2))

        return sample1, subsample1, subsample2

    print('train_length:', len(sample1))
    print('test_length:', len(sample2))

    return sample1, sample2


def create_or_sample_datasets(config: Configuration, sampled=True, save_path='data/'):
    """
    processes and / or samples datasets from config (datasets)

    :param config: (config.Configuration) configuration of datasets (filenames + sampling size)
    :param sampled: (bool) whether to sample (True) or not (False)
    :param save_path: (str) path to save final datasets to
    :return: list of datasets
    """
    subsets = []
    datasets = config.datasets

    for dataset in datasets:
        print(f'_______________SAMPLE FOR {dataset.name}____________________')
        print('Loading data...')

        # load annotation
        anno = read_anno(anno_file=dataset.file,
                         quote_file=dataset.quote,
                         columns=['lobenummer', 'ddo_dannetsemid', 'ddo_lemma', 'ddo_ordklasse', 'ddo_homnr',
                                  'ddo_definition', 'cor_bet_inventar', 'ddo_betyd_nr', 'citat', 'ddo_bemaerk',
                                  'cor_onto', 'cor_onto2', 'cor_hyper', 'cor_frame', 'ddo_konbet', 'ddo_encykl',
                                  'bow', 'length']
                         )

        # rename columns
        anno.columns = ['sense_id', 'ddo_dannetsemid', 'ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'ddo_definition',
                        'cor', 'ddo_betyd_nr', 'citat', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame', 'konbet',
                        'encykl', 'bow', 'length']

        print('Data loaded.')

        if sampled:  # sample data into train, test (and devel)
            if dataset.subsample_size:  # train, devel, test
                print('Sampling...')
                train, devel, test = sample(anno,
                                            dataset.sample_size,
                                            dataset.bias,
                                            subsample_size=dataset.subsample_size,
                                            sub_bias=dataset.sub_bias
                                            )
                train.to_csv(f'{save_path}/{dataset.name}_train.tsv', sep='\t', encoding='utf8')
                devel.to_csv(f'{save_path}/{dataset.name}_devel.tsv', sep='\t', encoding='utf8')
                test.to_csv(f'{save_path}/{dataset.name}_test.tsv', sep='\t', encoding='utf8')

                subsets += [(f"{dataset.name}_train", train),
                            (f"{dataset.name}_devel", devel),
                            (f"{dataset.name}_test", test)]

                print(f'Saved {dataset.name} (train {len(train)}, devel {len(devel)}, test {len(test)}) to data/')

            else:  # just train and test
                train, test = sample(anno, dataset.sample_size, dataset.bias)

                train.to_csv(f'{save_path}/{dataset.name}_train.tsv', sep='\t', encoding='utf8')
                test.to_csv(f'{save_path}/{dataset.name}_test.tsv', sep='\t', encoding='utf8')

                subsets += [(f"{dataset.name}_train", train),
                            (f"{dataset.name}_test", test)]

                print(f'Saved {dataset.name} (train {len(train)},  test {len(test)}) to data/')
        else:  # no sampling
            anno.to_csv(f'{save_path}/{dataset.name}_processed.tsv', sep='\t', encoding='utf8')
            subsets += [(f"{dataset.name}_processed", anno)]

    return subsets
