import pathlib
import sys
import argparse
import torch

from pycor.load_annotations.datasets import DataSet
from pycor.load_annotations.load_annotations import read_procssed_anno, read_anno, create_or_sample_datasets
from pycor.models.bert import BertSense
from pycor.models.word2vec import word2vec_model
from pycor.utils.save_load import load_obj
from config import load_config_from_json


def load_and_sample_datasets(dataset_config):
    datasets = create_or_sample_datasets(dataset_config)

    for subset in datasets:
        print(f"\n______________________{subset.upper()}____________________________")
        anno = read_procssed_anno(f"data/{subset}.tsv")

        print(f'Creating sense selection dataset for {subset}.')
        sense_selection = DataSet(anno, "sense_selection", sentence_type='all')
        sense_selection.to_tsv(f"var/BERT_dataset_{subset}.tsv")
        print(f'Creating BERT reduction dataset for {subset}.')
        bert_reduction = DataSet(anno, "bert_reduction", sentence_type='all')
        bert_reduction.to_tsv(f"data/reduction/reduction_{subset}.tsv")

        if 'mellem' not in subset:
            print(f'Creating (max 5) sense selection dataset for {subset}')
            sense_selection2 = DataSet(anno, "sense_selection", sentence_type='all', max_sense=5)
            sense_selection2.to_tsv(f"var/BERT_dataset_{subset}_less.tsv")
            print(f'Creating (max 5) BERT reduction dataset for {subset}')
            bert_reduction2 = DataSet(anno, "bert_reduction", sentence_type='all', max_sense=5)
            bert_reduction2.to_tsv(f"data/reduction/reduction_{subset}_less.tsv")
            print(f'Creating (max 5) text based dataset for {subset}')
            text_based2 = DataSet(anno, "textbased_only", max_sense=5)
            text_based2.to_tsv(f"data/reduction/reduction_word2vec_{subset}_less.tsv")
            print(f'Creating (max 5) rule based dataset for {subset}')
            rule_based2 = DataSet(anno, "rulebased_only", max_sense=5)
            rule_based2.to_tsv(f"data/base/reduction_score_{subset}_less.tsv")

        print(f'Creating text based dataset for {subset}')
        text_based = DataSet(anno, "textbased_only")
        text_based.to_tsv(f"data/reduction/reduction_word2vec_{subset}.tsv")
        print(f'Creating rule based dataset for {subset}')
        rule_based = DataSet(anno, "rulebased_only")
        rule_based.to_tsv(f"data/base/reduction_score_{subset}.tsv")


def load_feature_dataset(dataset_config, infotypes, models, save_sample='data/', save_final='var/',
                         sample=True):

    datasets = create_or_sample_datasets(dataset_config, sampled=sample, save_path=save_sample)

    for subset in datasets:
        anno = read_procssed_anno(f"{save_sample}{subset}.tsv")

        feature_dataset = DataSet(anno, "feature", infotypes=infotypes, embedding_type=models)
        feature_dataset.to_tsv(f"{save_final}{subset}_feature_dataset.tsv")


def generate_embeddings(filename, citat, models, save_path):
    print("Loading data...")
    anno = read_anno(anno_file=filename,
                     quote_file=citat,
                     keyword_file='',
                     annotated=True)

    print('Data loaded.')

    DataSet(anno,
            "generate_embeddings",
            embedding_type=models,
            output_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--sample", help="create and sample datasets and save to this path", type=pathlib.Path)
    parser.add_argument("-e", "--embed", help="embed and save to this path", type=pathlib.Path)
    parser.add_argument("-f", "--feature", help="create feature dataset and save to this path", type=pathlib.Path)
    parser.add_argument("-i", "--input_file", help="file to process", type=pathlib.Path)
    parser.add_argument("config_path", help="config path", type=pathlib.Path)

    args = parser.parse_args()

    config_path = args.config_path
    config = load_obj(config_path, load_json=True)

    config = load_config_from_json(config)


    if args.sample:
        load_and_sample_datasets(config)

    if 'embed':

        print('Loading word2vec model')
        model_path = config.models.word2vec
        word2vec = word2vec_model.load_word2vec_format(model_path,
                                                       fvocab=model_path + '.vocab',
                                                       binary=False)
        print('Loaded word2vec model')

        print('Loading BERT')
        bert_model = 'Maltehb/danish-bert-botxo'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        bert = BertSense.from_pretrained(bert_model)
        bert.load_tokenizer(bert_model)
        bert.load_checkpoint(config.models.bert)
        bert.to(device)

        models = {'bert': bert, 'word2vec': word2vec}

        if 'feature':
            infos = ['cosine', 'bert', 'onto', 'main_sense', 'figurative']
            load_feature_dataset("config_datasets",
                                 infos,
                                 models,
                                 save_sample=sys.argv[3],
                                 save_final=sys.argv[4])

        if 'embed':
            filename, citat = sys.argv[3], sys.argv[4]
            generate_embeddings(filename, citat, models, save_path=config['model_paths']['save_path'])
            # generate_embeddings(filename, None, word2vec, save_path=config['model_paths2']['save_path'])
