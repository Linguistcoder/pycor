import argparse
import pathlib
from typing import List, Dict, Optional

import torch

from config import load_config_from_json
from pycor.load_annotations.config import Configuration
from pycor.load_annotations.datasets import DataSet
from pycor.load_annotations.load_annotations import create_or_sample_datasets
from pycor.models.bert import BertSense
from pycor.models.word2vec import word2vec_model
from pycor.utils.save_load import load_obj


def load_and_sample_datasets(dataset_config, save_path):
    data = create_or_sample_datasets(dataset_config, save_path=save_path)

    for subset_name, anno in data:
        print(f"\n______________________{subset_name.upper()}____________________________")
        print(f'Creating sense selection dataset for {subset_name}.')
        sense_selection = DataSet(anno, "sense_selection", sentence_type='all')
        sense_selection.to_tsv(f"{save_path}BERT_dataset_{subset_name}.tsv")
        print(f'Creating BERT reduction dataset for {subset_name}.')
        bert_reduction = DataSet(anno, "bert_reduction", sentence_type='all')
        bert_reduction.to_tsv(f"{save_path}reduction_{subset_name}.tsv")

        if 'mellem' not in subset_name:
            print(f'Creating (max 5) sense selection dataset for {subset_name}')
            sense_selection2 = DataSet(anno, "sense_selection", sentence_type='all', max_sense=5)
            sense_selection2.to_tsv(f"{save_path}BERT_dataset_{subset_name}_less.tsv")
            print(f'Creating (max 5) BERT reduction dataset for {subset_name}')
            bert_reduction2 = DataSet(anno, "bert_reduction", sentence_type='all', max_sense=5)
            bert_reduction2.to_tsv(f"{save_path}reduction_{subset_name}_less.tsv")
            print(f'Creating (max 5) text based dataset for {subset_name}')
            text_based2 = DataSet(anno, "textbased_only", max_sense=5)
            text_based2.to_tsv(f"{save_path}reduction_word2vec_{subset_name}_less.tsv")
            print(f'Creating (max 5) rule based dataset for {subset_name}')
            rule_based2 = DataSet(anno, "rulebased_only", max_sense=5)
            rule_based2.to_tsv(f"{save_path}reduction_score_{subset_name}_less.tsv")

        print(f'Creating text based dataset for {subset_name}')
        text_based = DataSet(anno, "textbased_only")
        text_based.to_tsv(f"{save_path}reduction_word2vec_{subset_name}.tsv")
        print(f'Creating rule based dataset for {subset_name}')
        rule_based = DataSet(anno, "rulebased_only")
        rule_based.to_tsv(f"{save_path}reduction_score_{subset_name}.tsv")

    return data


def load_feature_dataset(dataset_config: Configuration, infotypes: List, embedding_models: Dict,
                         save_path: pathlib.Path = 'data/', data: Optional[List] = None):
    if not data:
        data = create_or_sample_datasets(dataset_config, sampled=False, save_path=save_path)

    for subset, anno in data:
        feature_dataset = DataSet(anno, "feature", infotypes=infotypes, embedding_type=embedding_models)
        feature_dataset.to_tsv(f"{save_path}{subset}_feature_dataset.tsv")


def generate_embeddings(dataset_config, embedding_models, save_path, data: Optional[List] = None):
    if not data:
        data = create_or_sample_datasets(dataset_config, sampled=False, save_path=save_path)

    for subset, anno in data:
        DataSet(anno,
                "generate_embeddings",
                embedding_type=embedding_models,
                output_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--sample", help="create and sample datasets and save to this path", type=pathlib.Path)
    parser.add_argument("-e", "--embed", help="embed and save to this path", type=pathlib.Path)
    parser.add_argument("-f", "--feature", help="create feature dataset and save to this path", type=pathlib.Path)
    parser.add_argument("config_path", help="config path", type=pathlib.Path)

    args = parser.parse_args()

    config_path = args.config_path
    config = load_obj(config_path, load_json=True)

    config = load_config_from_json(config)

    if args.sample:
        datasets = load_and_sample_datasets(config.datasets, save_path=args.sample)

    if args.embed or args.feature:

        if not config.models:
            raise AttributeError(config.models)

        models = {}
        infos = []
        if config.models.word2vec:
            print('Loading word2vec model')
            model_path = config.models.word2vec
            word2vec = word2vec_model.load_word2vec_format(str(model_path),
                                                           fvocab=str(model_path) + '.vocab',
                                                           binary=False)
            print('Loaded word2vec model')
            models['word2vec'] = word2vec
            infos.append('cosine')

        if config.models.bert_name:
            print('Loading BERT')
            bert_model = config.models.bert_name
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bert = BertSense.from_pretrained(bert_model)
            bert.load_tokenizer(bert_model)

            bert.to(device)
            models['bert'] = bert
            infos.append('bert')

        if args.embed:
            generate_embeddings(config.datasets, models, save_path=args.embed)

        if args.feature:
            if config.models.bert_checkpoint:
                bert.load_checkpoint(config.models.bert_checkpoint)
                bert.to(device)
                models['bert'] = bert
            print('Loading saved BERT')

            infos += ['onto', 'main_sense', 'figurative']

            load_feature_dataset(config.datasets,
                                 infos,
                                 models,
                                 save_path=args.feature)
