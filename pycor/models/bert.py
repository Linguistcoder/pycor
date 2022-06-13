from collections import namedtuple

import torch
import numpy as np
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pycor.load_annotations.load_with_bert import Sense_Selection_Data, SentDataset, collate_batch
from pycor.models.save_checkpoints import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertSense(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Tanh() #
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()


def forwardbase(model, batch, device):
    batch = tuple(tensor.to(device) for tensor in batch)
    # import pdb; pdb.set_trace()

    bert_out = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])

    # returns the last hidden layer of the classification token further processed by a Linear layer
    # and a Tanh activation function
    bert_out = model.dropout(bert_out[1])
    # linear = model.relu(model.linear(bert_out))
    # class_out = model.out(linear)
    class_out = model.out(bert_out)

    return class_out.squeeze(-1)


class BertSenseToken(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.activation = torch.nn.Tanh()
        self.linear = torch.nn.Linear(config.hidden_size * 2, 192)
        self.out = torch.nn.Linear(192, 1)
        self.init_weights()


def forward_token(model, batch, device):
    def get_token_repr_idx(input_ids: torch.tensor):
        batch_size = input_ids.shape[0]
        placement = input_ids == 31748
        token_idxs = placement.nonzero().transpose(1, 0)
        return [token_idxs[1][token_idxs[0] == b] for b in range(batch_size)]

    def get_repr_avg(output_hidden_states, token_idx):
        layers_hidden = output_hidden_states[4:-4]

        if isinstance(layers_hidden, tuple):
            layers_hidden = torch.mean(torch.stack(layers_hidden), axis=0)

        batch_size = layers_hidden.shape[0]
        hidden_token = [layers_hidden[b, token_idx[b][i] + 1:token_idx[b][j], :]
                        for b in range(batch_size) for i, j in [(0, 1), (-2, -1)]
                        ]

        hidden_token = torch.stack([torch.mean(hidden, dim=0)
                                    if hidden.shape[0] > 1 else hidden.squeeze(0)
                                    for hidden in hidden_token]).reshape(batch_size, -1, 768)

        if hidden_token.shape[1] > 2:
            hidden_token_1 = hidden_token[:, 0]
            hidden_token_2 = hidden_token[:, -1]
            hidden_token = torch.concat((hidden_token_1, hidden_token_2), dim=0)

        return hidden_token.reshape(batch_size, 768 * 2)

    batch = tuple(tensor.to(device) for tensor in batch)
    # import pdb; pdb.set_trace()

    bert_out = model.bert(input_ids=batch[0],
                          attention_mask=batch[1],
                          token_type_ids=batch[2],
                          output_hidden_states=True
                          )

    hidden_states = bert_out.hidden_states
    token_ids = get_token_repr_idx(batch[0])
    new_output = get_repr_avg(hidden_states, token_ids)

    # returns the last hidden layer of the classification token further processed by a Linear layer
    # and a Tanh activation function
    # bert_out = model.dropout(bert_out[1])
    bert_out = model.dropout(new_output)
    linear = model.linear(model.activation(bert_out))
    # class_out = model.out(linear)
    class_out = model.out(linear)

    return class_out.squeeze(-1)


def forward_token_cos(model, batch, device):
    def get_token_repr_idx(input_ids: torch.tensor):
        batch_size = input_ids.shape[0]
        placement = input_ids == 31748
        token_idxs = placement.nonzero().transpose(1, 0)
        return [token_idxs[1][token_idxs[0] == b] for b in range(batch_size)]

    def get_repr_avg(output_hidden_states, token_idx, layers):
        layers_hidden = output_hidden_states[layers]

        if isinstance(layers_hidden, tuple):
            layers_hidden = torch.mean(torch.stack(layers_hidden), axis=0)

        batch_size = layers_hidden.shape[0]
        hidden_token = [layers_hidden[b, token_idx[b][i] + 1:token_idx[b][j], :]
                        for b in range(batch_size) for i, j in [(0, 1), (-2, -1)]
                        ]

        hidden_token = torch.stack([torch.mean(hidden, dim=0)
                                    if hidden.shape[0] > 1 else hidden.squeeze(0)
                                    for hidden in hidden_token])

        if hidden_token.shape[0] > 4:
            hidden_token_1 = hidden_token[:2]
            hidden_token_2 = hidden_token[-2:]
            hidden_token = torch.concat((hidden_token_1, hidden_token_2), dim=1)

        return hidden_token.reshape(batch_size, 2, 768)

    batch = tuple(tensor.to(device) for tensor in batch)
    # import pdb; pdb.set_trace()

    bert_out = model.bert(input_ids=batch[0],
                          attention_mask=batch[1],
                          token_type_ids=batch[2],
                          output_hidden_states=True
                          )

    bert_out = model.dropout(bert_out)
    hidden_states = bert_out.hidden_states
    token_ids = get_token_repr_idx(batch[0])
    new_output = get_repr_avg(hidden_states, token_ids, layers=-1)
    class_out = model.cos(new_output[:, 0, :], new_output[:, 1, :])

    return class_out


def get_model_and_tokenizer(model_name, model_type, device, checkpoint=False):
    # if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    # torch.distributed.barrier()

    config = BertConfig.from_pretrained(model_name, num_labels=2)

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    if model_type == 'bert_token':
        model = BertSenseToken.from_pretrained(model_name, config=config)
        forward = forward_token
    elif model_type == 'bert_token_cos':
        model = BertSenseToken.from_pretrained(model_name, config=config)
        forward = forward_token_cos
    else:
        model = BertSense.from_pretrained(model_name, config=config)
        forward = forwardbase

    # add new special token
    if '[TGT]' not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
        assert '[TGT]' in tokenizer.additional_special_tokens
        model.resize_token_embeddings(len(tokenizer))

    if checkpoint and checkpoint != 'None':
        model, loss = load_checkpoint(checkpoint, model, device)

    model.to(device)

    return model, tokenizer, forward


BERT_MODEL = 'Maltehb/danish-bert-botxo'
MODEL, TOKENIZER, FORWARD = get_model_and_tokenizer('Maltehb/danish-bert-botxo',
                                                    'bertbase',# 'bert_token_cos'
                                                     device,
                                                     checkpoint=r'pycor\models\checkpoints\model_0.pt')
#'/content/drive/MyDrive/SPECIALE/data/model_0.pt')


def get_BERT_score(data, model=MODEL, tokenizer=TOKENIZER, forward=FORWARD):
    reduction = SentDataset(Sense_Selection_Data(data, tokenizer))
    dataloader = DataLoader(reduction,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_batch)

    nb_eval_steps = 0
    score = []

    iterator = tqdm(dataloader, desc="Iteration")

    for batches in iterator:
        if len(batches) < 1:
            continue

        model.eval()
        with torch.no_grad():
            # print(batches)

            labels = batches[0][5]  # if isinstance(model, BertSense) else [b[1] for b in batches]

            # run model
            for batch in batches:
                logits = forward(model, batch[2:], device)

                logits = model.sigmoid(torch.arctan(logits))
                score.append(logits)

        nb_eval_steps += 1

    data['score'] = torch.tensor(score)

    return data

def get_bert_embedding(row, model=MODEL, tokenizer=TOKENIZER):
    bert_data = SentDataset(Sense_Selection_Data(row, tokenizer, data_type='single'))

    model.eval()
    with torch.no_grad():
        for batch in bert_data:
            bert_out = model.bert(input_ids=torch.tensor(batch.input_ids).to(device).unsqueeze(0),
                                  attention_mask=torch.tensor(batch.input_mask).to(device).unsqueeze(0),
                                  token_type_ids=torch.tensor(batch.segment_ids).to(device).unsqueeze(0)
                                  )

            embedding = bert_out[1].squeeze().detach().cpu().numpy()

    return embedding
