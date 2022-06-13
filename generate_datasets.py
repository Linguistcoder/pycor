import sys

import torch
from gensim.models import KeyedVectors
from pycor.load_annotations.datasets import DataSet
from pycor.load_annotations.load_annotations import read_procssed_anno, read_anno, create_sampled_datasets
from pycor.models.bert import get_model_and_tokenizer
from pycor.utils.save_load import load_obj


def load_and_create_datasets(config_name, config_path):
    config = load_obj(config_name, load_json=True, path=config_path)

    datasets = create_sampled_datasets(config)

    for subset in datasets:
        print(f"______________________{subset.upper()}____________________________")
        anno = read_procssed_anno(f"data/{subset}.tsv")

        sense_selection = DataSet(anno, "sense_selection", sentence_type='all')
        sense_selection.to_tsv(f"var/BERT_dataset_{subset}.tsv")
        bert_reduction = DataSet(anno, "bert_reduction", sentence_type='all')
        bert_reduction.to_tsv(f"data/reduction/reduction_{subset}.tsv")

        if 'mellem' not in subset:
            sense_selection2 = DataSet(anno, "sense_selection", sentence_type='all', max_sense=5)
            sense_selection2.to_tsv(f"var/BERT_dataset_{subset}_less.tsv")
            bert_reduction2 = DataSet(anno, "bert_reduction", sentence_type='all', max_sense=5)
            bert_reduction2.to_tsv(f"data/reduction/reduction_{subset}_less.tsv")
            text_based2 = DataSet(anno, "textbased_only", max_sense=5)
            text_based2.to_tsv(f"data/reduction/reduction_word2vec_{subset}_less.tsv")
            rule_based2 = DataSet(anno, "rulebased_only", max_sense=5)
            rule_based2.to_tsv(f"data/base/reduction_score_{subset}_less.tsv")

        text_based = DataSet(anno, "textbased_only")
        text_based.to_tsv(f"data/reduction/reduction_word2vec_{subset}.tsv")
        rule_based = DataSet(anno, "rulebased_only")
        rule_based.to_tsv(f"data/base/reduction_score_{subset}.tsv")

        # feature_based


def generate_embeddings(filename, bert, word2vec, save_path):
    print("Loading data...")
    anno = read_anno(anno_file=filename,
                     quote_file='',
                     keyword_file='',
                     annotated=True)

    print('Data loaded.')

    anno = DataSet(anno, "generate_embeddings", embedding_type={"bert": bert,
                                                                "word2vec": word2vec})

    anno.to_csv(f'{save_path}/annotations_with_embeddings.tsv', sep='\t', encoding='utf8')


if __name__ == "__main__":
    config_name = "config_model"
    config_path = "configs/"

    config = load_obj(config_name, load_json=True, path=config_path)

    print('Loading word2vec model')
    model_path = config['model_paths']['word2vec']
    word2vec = KeyedVectors.load_word2vec_format(model_path,
                                                 fvocab=model_path + '.vocab',
                                                 binary=False)
    print('Loaded word2vec model')

    print('Loading BERT')
    bert_model = 'Maltehb/danish-bert-botxo'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, forward = get_model_and_tokenizer('Maltehb/danish-bert-botxo',
                                                        'bertbase',  # 'bert_token_cos'
                                                        device,
                                                        checkpoint=config['model_paths']['bert'])

    # load_and_create_datasets(config_name, config_path)
    filename = sys.argv[1]  # 'data/hum_anno/all_09_06_2022.txt'
    generate_embeddings(filename, (model, tokenizer), word2vec, save_path=config['model_paths']['save_path'])
