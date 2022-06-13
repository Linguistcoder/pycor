import pandas as pd
from pycor.utils.preprocess import clean_str, remove_special_char
import random

random.seed(81)

def read_anno(anno_file: str, quote_file: str, keyword_file: str, columns=None, annotated=False) -> pd.DataFrame:

    if columns is None:
        columns = []

    anno = pd.read_csv(anno_file,
                       sep='\t',
                       encoding='utf8',
                       na_values=['n', ' '],
                       index_col=False
                       )
    if annotated is True:
        anno = anno.dropna(subset=['ddo_lemma', 'cor_bet_inventar', 'ddo_dannetsemid'])
        anno = anno[anno['cor_bet_inventar'] != '0']
    else:
        anno = anno.dropna(subset=['ddo_lemma', 'ddo_dannetsemid'])

    anno['ddo_dannetsemid'] = anno['ddo_dannetsemid'].astype('int64')
    anno['ddo_homnr'] = anno['ddo_homnr'].fillna(1)

    bow = ['ddo_lemma', 'ddo_definition', 'ddo_genprox']

    if quote_file:
        quotes = pd.read_csv(quote_file,
                             sep='\t',
                             encoding='utf-8',
                             usecols=['ddo_dannetsemid', 'citat']
                             )
        quotes = quotes.groupby('ddo_dannetsemid').aggregate('||'.join)
        anno = anno.merge(quotes, how='outer', on=['ddo_dannetsemid'])

        #anno.citat = anno.citat.apply(clean_str)

        anno.citat2 = anno.citat.apply(lambda x: x.replace('[TGT]', '') if isinstance(x, str) else '')
        anno.citat2 = anno.citat2.apply(lambda x: x.replace('||', '') if isinstance(x, str) else '')

    if keyword_file:
        keywords = pd.read_csv(keyword_file,
                               sep='\t',
                               encoding='utf8')
        anno = anno.merge(keywords, how='outer', on=['ddo_dannetsemid'])


    if 'ddb_afsnit' in anno and 'ddb_stikord_brutto' in anno:
        anno["ddb_stikord"] = anno.ddb_stikord_brutto.apply(lambda x: x.replace(',', ' ').lower() if isinstance(x, str) else '')
        anno.ddb_stikord = anno.ddb_stikord.apply(remove_special_char)
        anno.ddb_afsnit = anno.ddb_afsnit.apply(lambda x: x.replace(';', ' ').lower() if isinstance(x, str) else '')
        anno.ddb_afsnit = anno.ddb_afsnit.apply(remove_special_char)
        bow += ['ddb_afsnit', 'ddb_stikord']


    anno = anno.dropna(subset=['ddo_lemma', 'ddo_genprox', 'ddo_dannetsemid', 'ddo_betyd_nr']) # 'COR_bet_inventar'

    anno.ddo_konbet = anno.ddo_konbet.fillna('').astype(str)
    anno.ddo_encykl = anno.ddo_encykl.fillna('').astype(str)
    anno.ddo_definition = anno.ddo_definition.fillna('').astype(str)

    #anno.citat = anno.citat.apply(lambda x: x.replace('[TGT]', '') if isinstance(x, str) else [])
    anno.ddo_definition = anno[['ddo_definition', 'ddo_konbet', 'ddo_encykl']].aggregate(' '.join, axis=1)

    anno.ddo_definition = anno.ddo_definition.apply(remove_special_char)

    anno['bow'] = anno[bow].aggregate(' '.join, axis=1)
    anno['length'] = anno.bow.apply(lambda x: len([i for i in x.split('||') if i != '']))

    #T = anno.length.sum()
    #all_words = [i for i in '||'.join(anno.bow.values).split('||') if i != '']
    #uniq_words = set([i for i in all_words if i != ''])
    #freq_dict = {word: all_words.count(word) for word in uniq_words}
    #print(len(uniq_words))

    anno = anno.fillna('')
    #anno['onto'] = anno[['dn_onto1', 'dn_onto2', 'COR-onto']].aggregate('+'.join, axis=1)
    #anno = anno[anno.length > 0]
    if len(columns):
        anno = anno.loc[:, columns]

    return anno  # , T, freq_dict

# anno.merge(citater,
#           how='outer',
#           on='ddo_dannetsemid'
#           ).loc[:, ['ddo_dannetsemid', 'ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'citat']].to_csv('var/citater.tsv',
#                                                                                 sep='\t',
#                                                                                 encoding='utf8')

def read_procssed_anno(anno_file: str) -> pd.DataFrame:
    anno = pd.read_csv(anno_file,
                       sep='\t',
                       encoding='utf8',
                       na_values=[' ']
                       )

    return anno


def sample(population, sample_size, bias, **subgroup):
    lemma_groups = population.groupby(['lemma', 'ordklasse', 'homnr'])
    groups = [idx for idx in lemma_groups.groups.values() if len(idx) > 1]

    n_samples = int(len(groups) * sample_size)
    sampler = random.sample(groups, k=n_samples + bias)
    indicies = [i for idx in sampler for i in idx]

    sample1 = population[population.index.isin(indicies)]
    sample2 = population[~population.index.isin(indicies)]

    if subgroup:
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


def create_sampled_datasets(datasets: dict):
    subsets = []
    for dataset, config in datasets.items():
        print(f'_______________SAMPLE FOR {dataset}____________________')
        print('Loading data...')
        anno = read_anno(anno_file=config['file'],
                         quote_file=config['quote'],
                         keyword_file=config['keyword'],
                         columns=['ddo_lemma', 'ddo_ordklasse', 'ddo_homnr', 'ddo_definition', 'ddo_genprox',
                                  'ddo_kollokation',
                                  'cor_bet_inventar', 'dn_id', 'ddo_betyd_nr', 'citat', 'ddo_bemaerk', 'dn_onto1',
                                  'dn_onto2',
                                  'dn_hyper', 'frame', 'ddo_konbet', 'ddo_encykl', 'bow', 'length']
                         )

        anno.columns = ['lemma', 'ordklasse', 'homnr', 'definition', 'genprox', 'kollokation', 'cor', 'dn_id', 'ddo_nr',
                        'citat', 'bemaerk', 'onto1', 'onto2', 'hyper', 'frame', 'konbet', 'encykl', 'bow', 'length']

        print('Data loaded.')

        if 'subsample_size' in config:
            print('Sampling...')
            train, devel, test = sample(anno,
                                        config['sample_size'],
                                        config['bias'],
                                        subsample_size=config['subsample_size'],
                                        sub_bias=config['sub_bias']
                                        )
            train.to_csv(f'data/{dataset}_train.tsv', sep='\t', encoding='utf8')
            devel.to_csv(f'data/{dataset}_devel.tsv', sep='\t', encoding='utf8')
            test.to_csv(f'data/{dataset}_test.tsv', sep='\t', encoding='utf8')

            subsets += [f"{dataset}_train", f"{dataset}_devel", f"{dataset}_test"]

            print(f'Saved {dataset} (train {len(train)}, devel {len(devel)}, test {len(test)}) to data/')

        else:
            train, test = sample(anno, config['sample_size'], config['bias'])

            train.to_csv(f'data/{dataset}_train.tsv', sep='\t', encoding='utf8')
            test.to_csv(f'data/{dataset}_test.tsv', sep='\t', encoding='utf8')

            subsets += [f"{dataset}_train", f"{dataset}_test"]

            print(f'Saved {dataset} (train {len(train)},  test {len(test)}) to data/')

    return subsets
