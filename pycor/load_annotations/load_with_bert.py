from collections import namedtuple
from typing import List
import torch


class Sense_Selection_Data(List):
    def __init__(self, data, tokenizer, max_seq_length=254, pad_token=0,
                 mask_zero_padding=True,
                 emb_model=None, data_type='reduction'):
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.emb_model = emb_model
        self.mask_zero_padding = mask_zero_padding

        if data_type == 'reduction':
            super().__init__(self.load_reduction_data(data, tokenizer))
        else:
            super().__init__(self.single_row(data, tokenizer))

    def truncate_pair_to_max_length(self, tokens):
        first_sep = tokens.token_type_ids.index(1)
        tokens_a = {'input_ids': tokens.input_ids[:first_sep - 1],
                    'token_type_ids': tokens.token_type_ids[:first_sep - 1],
                    'attention_mask': tokens.attention_mask[:first_sep - 1]}

        tokens_b = {'input_ids': tokens.input_ids[first_sep:-1],
                    'token_type_ids': tokens.token_type_ids[first_sep:-1],
                    'attention_mask': tokens.attention_mask[first_sep:-1]}

        total_length = len(tokens_a['input_ids']) + len(tokens_b['input_ids'])

        while total_length > self.max_seq_length - 2:
            total_length = len(tokens_a['input_ids']) + len(tokens_b['input_ids'])

            if total_length <= self.max_seq_length - 2:
                tokens.input_ids = tokens_a['input_ids'] + [3] + tokens_b['input_ids'] + [3]
                tokens.token_type_ids = tokens_a['token_type_ids'] + [0] + tokens_b['token_type_ids'] + [1]
                tokens.attention_mask = tokens_a['attention_mask'] + [1] + tokens_b['attention_mask'] + [1]
                return tokens

            if len(tokens_a) > len(tokens_b):
                tokens_a['input_ids'].pop()
                tokens_a['token_type_ids'].pop()
                tokens_a['attention_mask'].pop()
            else:

                tokens_b['input_ids'].pop()
                tokens_b['token_type_ids'].pop()
                tokens_b['attention_mask'].pop()

    def load_reduction_data(self, data, tokenizer):
        datapoints = []
        BertInput = namedtuple("BertInput", ["lemma", "row",
                                             "input_ids", "attention_mask",
                                             "segment_ids", "label_id"])

        for row in data.itertuples():
            pairs = []
            tokens = tokenizer(row.sentence_1, row.sentenc2)

            if len(tokens.input_ids) > self.max_seq_length:
                tokens = self.truncate_pair_to_max_length(tokens)

            # Zero pad to the max_seq_length.
            pad_length = self.max_seq_length - len(tokens.input_ids)

            input_ids = tokens.input_ids + ([self.pad_token] * pad_length)
            attention_mask = tokens.attention_mask + ([0 if self.mask_zero_padding else 1] * pad_length)
            segment_ids = tokens.token_type_ids + ([0] * pad_length)

            assert len(input_ids) == self.max_seq_length
            assert len(attention_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            pairs.append(BertInput(lemma=row.lemma,
                                   row=row.Index,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   segment_ids=segment_ids,
                                   label_id=row.label))

            datapoints.append(pairs)

        return datapoints

    def single_row(self, row, tokenizer):
        BertInput = namedtuple("BertInput", ["lemma", "onto", "COR", "DDO",
                                             "wcl", "input_ids", "attention_mask",
                                             "segment_ids"])
        datapoints = []

        tokens = tokenizer(row.citat)

        if len(tokens.input_ids) > self.max_seq_length:
            tokens.input_ids = tokens.input_ids[:self.max_seq_length - 1] + [3]
            tokens.token_type_ids = tokens.token_type_ids[:self.max_seq_length - 1] + [0]
            tokens.attention_mask = tokens.attention_mask[:self.max_seq_length - 1] + [1]

        # Zero pad to the max_seq_length
        pad_length = self.max_seq_length - len(tokens.input_ids)

        input_ids = tokens.input_ids + ([self.pad_token] * pad_length)
        attention_mask = tokens.attention_mask + ([0 if self.mask_zero_padding else 1] * pad_length)
        segment_ids = tokens.token_type_ids + ([0] * pad_length)

        assert len(input_ids) == self.max_seq_length
        assert len(attention_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        datapoints.append(BertInput(lemma=row.ddo_lemma,
                                    onto=row.cor_onto,
                                    COR=row.cor_bet_inventar,
                                    DDO=row.ddo_betyd_nr,
                                    wcl=row.ddo_ordklasse,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    segment_ids=segment_ids))

        return datapoints


def collate_batch(batch):
    max_seq_length = len(batch[0][0].input_ids)

    collated = []
    for sub_batch in batch:
        batch_size = len(sub_batch)

        id_collated = torch.zeros([batch_size, max_seq_length], dtype=torch.long)
        mask_collated = torch.zeros([batch_size, max_seq_length], dtype=torch.long)
        segment_collated = torch.zeros([batch_size, max_seq_length], dtype=torch.long)
        label_collated = torch.zeros([batch_size], dtype=torch.long)

        for i, bert_input in enumerate(sub_batch):
            id_collated[i] = torch.tensor(bert_input.input_ids, dtype=torch.long)
            mask_collated[i] = torch.tensor(bert_input.input_mask, dtype=torch.long)
            segment_collated[i] = torch.tensor(bert_input.segment_ids, dtype=torch.long)
            label_collated[i] = torch.tensor(bert_input.label_id, dtype=torch.long)

        collated.append([bert_input.lemma,
                         bert_input.row,
                         id_collated, mask_collated, segment_collated, label_collated])

    return collated


class SentDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
