import pandas as pd
import torch

def to_tensor(string):
    string = string.strip('tensor(')
    string = string.strip(')')
    embedding = eval(string)
    return torch.tensor(embedding)

def read_embeddings_file(filename):
    file = pd.read_csv(filename,
                       sep='\t',
                       encoding='utf8')

    file['embeddings'] = file['embeddings'].apply(to_tensor)

    return file

embeddings = read_embeddings_file(r'C:\Users\nmp828\Documents\pycor\var\reduction_emb_cbc_devel2.tsv')

