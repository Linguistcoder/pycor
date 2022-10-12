from collections import namedtuple
from typing import List
import torch


class Sense_Selection_Data(List):
    """
    class for creating a sense selection dataset.

    Creates either an entire reduction dataset (data_type='reduction') or handles a single sample (data_type='single')

    A sense selection dataset pairs a target sense context sentence with a sample of other contexts for other senses
    of the target lemma from a sense inventory. If the paired contexts come from the same sense in the sense inventory,
    then the label == 1.

    Attributes
    ----------
    :attr max_seq_length: maximum sequence length to truncate to
    :attr pad_token: ID to use as padding token
    :attr emb_model: the BERT embedding model to use (BertSense or BertSenseToken)
    :attr mask_zero_padding: whether to mask zero padding

    Methods
    -------
    truncate_pair_to_max_length(self, tokens)
        :returns: truncated tokens for sentence_1 + sentence_2

    load_reduction_data(self, data, tokenizer)
        :returns: list of bert_input data points

    single_row(self, data, tokenizer)
        :returns: list with a single bert_input

    """

    def __init__(self, data, tokenizer, max_seq_length=254, pad_token=0, mask_zero_padding=True,
                 emb_model=None, data_type='reduction'):
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.emb_model = emb_model
        self.mask_zero_padding = mask_zero_padding

        if data_type == 'reduction':  # full reduction dataset
            super().__init__(self.load_reduction_data(data, tokenizer))
        else:  # single sample for when we just want to retrieve the embedding or a single BERT score
            super().__init__(self.single_row(data, tokenizer))

    def truncate_pair_to_max_length(self, tokens):
        """truncates a sentence pair (sentence_1 + sentence_2) to self.max_seq_length"""
        # split input into sentence_1 (tokens_a) and sentence_2 (tokens_b)
        first_sep = tokens.token_type_ids.index(1)  # index(1) gives the start index of sentence_2
        tokens_a = {'input_ids': tokens.input_ids[:first_sep - 1],  # slice to first_step-1 as we don't count [SEP]
                    'token_type_ids': tokens.token_type_ids[:first_sep - 1],
                    'attention_mask': tokens.attention_mask[:first_sep - 1]}

        tokens_b = {'input_ids': tokens.input_ids[first_sep:-1],  # slice to -1 as the last token is also [SEP]
                    'token_type_ids': tokens.token_type_ids[first_sep:-1],
                    'attention_mask': tokens.attention_mask[first_sep:-1]}

        total_length = len(tokens_a['input_ids']) + len(tokens_b['input_ids'])

        while total_length > self.max_seq_length - 2:
            total_length = len(tokens_a['input_ids']) + len(tokens_b['input_ids'])

            if total_length <= self.max_seq_length - 2:
                # combine sentences + add [SEP]s again
                tokens.input_ids = tokens_a['input_ids'] + [3] + tokens_b['input_ids'] + [3]
                tokens.token_type_ids = tokens_a['token_type_ids'] + [0] + tokens_b['token_type_ids'] + [1]
                tokens.attention_mask = tokens_a['attention_mask'] + [1] + tokens_b['attention_mask'] + [1]
                return tokens

            if len(tokens_a['input_ids']) > len(tokens_b['input_ids']):
                # if the first sentence is the longest --> remove the last token of that sequence
                tokens_a['input_ids'].pop()
                tokens_a['token_type_ids'].pop()
                tokens_a['attention_mask'].pop()
            else:
                # if the second sentence is the longest --> remove the last token of that sequence
                tokens_b['input_ids'].pop()
                tokens_b['token_type_ids'].pop()
                tokens_b['attention_mask'].pop()

    def load_reduction_data(self, data, tokenizer):
        """
        creates a reduction dataset based on a SenseSelection Framework
        :param data: (pd.DataFrame) with columns [lemma, sentence_1, sense_1_id, sentence_2, sense_2_id, label])
        :param tokenizer: Bert Tokenizer
        :return: (list) of bert_input data points
        """
        datapoints = []
        bert_input = namedtuple("bert_input", ["lemma", "row",
                                               "input_ids", "attention_mask",
                                               "segment_ids", "label_id"])

        for row in data.itertuples():
            pairs = []

            # each sentence needs to be at least two tokens long
            if len(row.sentence_1) < 2 or len(row.sentence_2) < 2:
                continue

            tokens = tokenizer(row.sentence_1, row.sentence_2)

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

            pairs.append(bert_input(lemma=row.lemma,
                                    row=row.Index,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    segment_ids=segment_ids,
                                    label_id=row.label))

            datapoints.append(pairs)

        return datapoints

    def single_row(self, row, tokenizer, quote=False):
        """
        creates a single sample reduction "dataset" for a single sentence
        :param row: (Pandas NamedTuple) with columns
                    [ddo_lemma, cor_onto, cor_bet_inventar, ddo_betyd_nr, ddo_ordklasse, ddo_definition]
        :param tokenizer: Bert Tokenizer
        :param quote: whether to use quotes (True) or definitions (False). Defaults to definitions.
        :return:
        """
        bert_input = namedtuple("bert_input", ["lemma", "onto", "COR", "DDO",
                                               "wcl", "input_ids", "attention_mask",
                                               "segment_ids"])
        datapoints = []

        if quote:
            # multiple quotes possible
            tokens = []
            for citat in row.citat.split('||'):
                if len(tokens) > 1:  # only use the first quote
                    break
                tokens = tokenizer(citat)
                if len(tokens.input_ids) < 3:  # ignore empty quotes
                    continue
        else:
            tokens = tokenizer('[TGT]' + row.ddo_lemma + '[TGT]' + row.ddo_definition)

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

        datapoints.append(bert_input(lemma=row.ddo_lemma,
                                     onto=row.cor_onto,
                                     COR=row.cor_bet_inventar,
                                     DDO=row.ddo_betyd_nr,
                                     wcl=row.ddo_ordklasse,
                                     input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     segment_ids=segment_ids))

        return datapoints


def collate_batch(batch):
    """same sequence length for entire batch"""
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
            mask_collated[i] = torch.tensor(bert_input.attention_mask, dtype=torch.long)
            segment_collated[i] = torch.tensor(bert_input.segment_ids, dtype=torch.long)
            label_collated[i] = torch.tensor(bert_input.label_id, dtype=torch.long)

        collated.append([bert_input.lemma,
                         bert_input.row,
                         id_collated, mask_collated, segment_collated, label_collated])

    return collated


class SentDataset(torch.utils.data.Dataset):
    """torch dataset class for sense selection data"""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
