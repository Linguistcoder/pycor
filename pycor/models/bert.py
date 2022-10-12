import numpy as np
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
import torch
from torch.utils.data import DataLoader

from pycor.load_annotations.load_with_bert import Sense_Selection_Data, SentDataset, collate_batch


def get_token_repr_idx(input_ids: torch.tensor):
    """get the placement of [TGT] token in the bert input"""
    batch_size = input_ids.shape[0]
    placement = input_ids == 31748  # todo: make variable
    token_idxs = placement.nonzero().transpose(1, 0)
    return [token_idxs[1][token_idxs[0] == b] for b in range(batch_size)]


def get_repr_avg(output_hidden_states, token_idx, n_sent=1):
    """calculate the average representation for the tokens in range token_idx for the last four layers.
    if multiple tokens are in range token_idx, then the final representation is:
        average of the four last layers --> average of tokens --> final representation

    :param output_hidden_states: Bert output
    :param token_idx: (list) start and end index for target tokens
    :param n_sent: (int) 1 sentence or 2 sentence input
    """
    layers_hidden = output_hidden_states[-4:]  # get four last hidden layers
    layers_hidden = torch.mean(torch.stack(layers_hidden), axis=0)  # first average (hidden layers)

    batch_size = layers_hidden.shape[0]
    hidden_token = [layers_hidden[b, token_idx[b][0]:token_idx[b][1] - 1, :]
                    for b in range(batch_size)]  # get target tokens for each instance in batch

    hidden_token = torch.stack([torch.mean(hidden, dim=0)
                                if hidden.shape[0] > 1 else hidden.squeeze(0)
                                for hidden in hidden_token]).reshape(batch_size, -1, 768)  # second average (token)

    if hidden_token.shape[0] > n_sent * 2:  # if multiple [TGT] are present, then only use the first one
        hidden_token_1 = hidden_token[:2]
        hidden_token_2 = hidden_token[-2:]
        hidden_token = torch.concat((hidden_token_1, hidden_token_2), dim=1)

    # return hidden_token.reshape(batch_size, 2, 768)
    return hidden_token.reshape(batch_size, n_sent * 768)


class BERT(BertPreTrainedModel):
    """
    BERT model with extra methods to save/load model and retrieve the BERT output.
    Inherits from BertPreTrainedModel

    Attributes
    ----------
    :attr num_labels: number of labels for training
    :attr config: bert config
    :attr device_attr: GPU device (if available)
    :attr bert: BertModel

    Methods
    -------
    load_tokenizer(self, name)
        :returns: tokenizer loaded from name with added [TGT] token

    save_checkpoint(self, path, valid_loss)
        :returns: None (saves model to path)

    load_checkpoint(self, path)
        :returns: updated state dict (load from path)

    save_metrics(self, path):
        :returns: None (saves metrics to path)

    load_metrics(self, path)
       :returns: updated state dict (load metrics from path)

    get_BERT_score(self, data, output_type='data')
        :returns: dataset with bert output score or just the output scores

    get_bert_embedding(self, row, token=True)
        :returns: single bert embedding as list for data in row
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.device_attr = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BertModel(config)

        self.post_init()

    def load_tokenizer(self, name):
        """loads the tokenizer from name via HuggingFace and adds a special target token [TGT]"""
        tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=True)

        # add new special token
        if '[TGT]' not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
            assert '[TGT]' in tokenizer.additional_special_tokens
            self.resize_token_embeddings(len(tokenizer))

        self.tokenizer = tokenizer
        return tokenizer

    def save_checkpoint(self, path, valid_loss):
        """save model checkpoint to path"""
        state_dict = {'model_state_dict': self.state_dict(),
                      'valid_loss': valid_loss}

        torch.save(state_dict, path)
        print(f'Saved model to: {path}')

    def load_checkpoint(self, path):
        """load model checkpoint from path.
        :returns: updated model state dict
        """
        state_dict = torch.load(path, map_location=self.device)
        print(f'Loaded model from: {path}')
        print(state_dict.keys())
        self.load_state_dict(state_dict['model_state_dict'])
        return state_dict['valid_loss']

    def save_metrics(self, path, train_loss_list, valid_loss_list, global_steps_list):
        """save model metrics to path"""
        state_dict = {'train_loss_list': train_loss_list,
                      'valid_loss_list': valid_loss_list,
                      'global_steps_list': global_steps_list}

        torch.save(state_dict, path)
        print(f'Saved model to: {path}')

    def load_metrics(self, path):
        """load model metrics from path.
        :returns: updated model state dict
        """
        state_dict = torch.load(path, map_location=self.device)
        print(f'Loaded model from: {path}')

        return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

    def get_BERT_score(self, data, output_type='data'):
        """
        Get the BERT score for each instance in data.
        The BERT score is an estimation of the similarity / proximity of the two sense inputs
        :param data: (DataFrame) with columns: [lemma, sentence_1, sense_1_id, sentence_2, sense_2_id, label])
        :param output_type: (str) whether to return DataFrame ('data') or Tensor of scores ('tensor')
        :return: pd.DataFrame with new ['score'] column or torch.tensor of scores
        """
        reduction = SentDataset(Sense_Selection_Data(data, self.tokenizer))  # transform data into a bert dataset
        dataloader = DataLoader(reduction,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=collate_batch)

        nb_eval_steps = 0
        score = []
        sigmoid = torch.nn.Sigmoid()

        for batches in dataloader:  # iterator:
            if len(batches) < 1:  # ignore empty instances
                continue

            self.eval()
            with torch.no_grad():
                # run model
                for batch in batches:
                    logits = self(input_ids=batch[2].to(self.device),
                                  attention_mask=batch[3].to(self.device),
                                  token_type_ids=batch[4].to(self.device)
                                  )

                    logits = sigmoid(torch.arctan(logits))  # score between 0-1
                    score.append(logits.cpu())  # score moved from GPU to CPU to save it

            nb_eval_steps += 1

        if output_type == 'data':  # return dataframe
            data['score'] = torch.tensor(score)
            return data
        else:  # return tensor
            return torch.tensor(score)

    def get_bert_embedding(self, row, token=True):
        """

        :param row: (Pandas NamedTuple) row with dictionary information
        :param token: whether to use token embeddings (True) or CLS embedding (False
        :return: (list) embedding
        """
        # transform row to BERT dataset. The sentence type (definition or quote) is determined in Sense_Selection_Data
        bert_data = SentDataset(Sense_Selection_Data(row, self.tokenizer, data_type='single'))

        # Only if sentence is in the data. This is relevant if we use quotes.
        if len(bert_data) < 1:
            return np.nan

        self.eval()
        with torch.no_grad():
            # run model
            for batch in bert_data:
                input_ids = torch.tensor(batch.input_ids)

                if token:
                    tgt_token_ids = get_token_repr_idx(input_ids.unsqueeze(0))  # get token placement

                    # input to bert without the special token [TGT] (it influences the embedding too much)
                    attention_mask = torch.tensor(batch.attention_mask)[input_ids != 31748]
                    segment_ids = torch.tensor(batch.segment_ids)[input_ids != 31748]
                    input_ids = input_ids[input_ids != 31748]

                    bert_out = self.bert(input_ids=input_ids.to(self.device).unsqueeze(0),
                                         attention_mask=attention_mask.to(self.device).unsqueeze(0),
                                         token_type_ids=segment_ids.to(self.device).unsqueeze(0),
                                         output_hidden_states=True
                                         )

                    hidden_states = bert_out.hidden_states
                    # retrieve the target token (placement known from [TGT])
                    output = get_repr_avg(hidden_states, tgt_token_ids)

                else:
                    # BERT input (we keep the special token [TGT] in this setup. We want to keep its influence here)
                    input_ids, attention_mask, segment_ids = batch.input_ids, batch.attention_mask, batch.segment_ids

                    bert_out = self.bert(input_ids=torch.tensor(input_ids).to(self.device).unsqueeze(0),
                                         attention_mask=torch.tensor(attention_mask).to(self.device).unsqueeze(0),
                                         token_type_ids=torch.tensor(segment_ids).to(self.device).unsqueeze(0)
                                         )

                    output = bert_out[1]

                # we need to resize and move the final embedding to CPU
                embedding = output.squeeze().detach().cpu().numpy()

        return embedding.tolist()


class BertSense(BERT):
    """
    BERT model for sense similarity / proximity estimation using CLS token
    Inherits from BERT

    Attributes
    ----------
    :attr num_labels: number of labels for training
    :attr config: bert config
    :attr dropout: dropout level
    :attr sigmoid: sigmoid activation function
    :attr relu: Leaky ReLU activation function
    :attr out: linear output layer
    :attr softmax: softmax function

    Methods
    -------
    forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None)
        :returns: Bert sense similarity / proximity score
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.relu = torch.nn.LeakyReLU()
        self.out = torch.nn.Linear(self.config.hidden_size, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None):
        """updated forward function for sense similarity / proximity estimation"""
        # import pdb; pdb.set_trace()
        bert_out = self.bert(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=True,
                             return_dict=return_dict
                             )
        # returns the last hidden layer of the classification token further processed by a relu activation function
        # and a linear layer
        bert_out = self.dropout(bert_out[1])
        class_out = self.out(self.relu(bert_out))
        class_out = self.softmax(class_out)

        return class_out.squeeze()[0]


class BertSenseToken(BERT):
    """
    BERT model for sense similarity / proximity estimation using average target token embedding
    Inherits from BERT

    Attributes
    ----------
    :attr num_labels: number of labels for training
    :attr config: bert config
    :attr dropout: dropout level
    :attr sigmoid: sigmoid activation function
    :attr cos: cosine similarity function
    :attr activation1: Leaky ReLU activation function for the first postprocessing layer
    :attr activation2: TanH actionvation function for the second postprocessing layer
    :attr reduce: first postprocessing linear layer that reduces nodes from hidden_size to 192
    :attr combine: second postprocessing linear layer that combines the two sentence embeddings
    :attr out: linear output layer

    Methods
    -------
    forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None)
        :returns: Bert sense similarity / proximity score using target token embedding

    forward_cos(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None)
        :returns: Bert sense similarity / proximity score using cosine similarity
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = torch.nn.CosineSimilarity(dim=1)

        self.activation1 = torch.nn.LeakyReLU()
        self.activation2 = torch.nn.Tanh()
        self.reduce = torch.nn.Linear(config.hidden_size, 192)
        self.combine = torch.nn.Linear(192 * 2, 192)
        self.out = torch.nn.Linear(192, 2)
        self.softmax = torch.nn.Softmax(dim=1)

        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None):
        """updated forward function for sense similarity / proximity estimation"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        token_ids = get_token_repr_idx(input_ids)  # get target token placement

        bert_out = self.bert(input_ids[input_ids != 31748],  # remove [TGT] token
                             attention_mask=attention_mask[input_ids != 31748],  # remove [TGT] token
                             token_type_ids=token_type_ids[input_ids != 31748],  # remove [TGT] token
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=True,
                             return_dict=return_dict
                             )

        hidden_states = bert_out.hidden_states
        # retrieve the target token (placement known from [TGT])
        new_output = get_repr_avg(hidden_states, token_ids, n_sent=2)

        bert_out = self.dropout(new_output)

        # returns the last hidden layer of the classification token further processed by a Linear layer
        # and a leaky ReLU activation function
        linear = self.reduce(self.activation1(bert_out))

        # combines the two token representations through a linear layer + a Tanh activation function
        linear = self.combine(self.activation2(linear))
        # class_out = model.out(linear)
        class_out = self.softmax(self.out(linear))

        return class_out.squeeze(-1)

    def forward_cos(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                    inputs_embeds=None, output_attentions=None, return_dict=None):
        """updated forward function using cosine similarity instead of trained postprocessing
        #####THIS HAS NOT BEEN TESTED#####
        """

        # import pdb; pdb.set_trace()
        token_ids = self.get_token_repr_idx(input_ids)  # get target token placement

        bert_out = self.bert(input_ids[input_ids != 31748],  # remove [TGT] token
                             attention_mask=attention_mask[input_ids != 31748],  # remove [TGT] token
                             token_type_ids=token_type_ids[input_ids != 31748],  # remove [TGT] token
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=True,
                             return_dict=return_dict
                             )

        bert_out = self.dropout(bert_out)
        hidden_states = bert_out.hidden_states
        # retrieve the target token (placement known from [TGT])
        new_output = self.get_repr_avg(hidden_states, token_ids)
        # reduce dimensionality
        new_output = self.reduce(self.activation1(new_output))
        class_out = self.cos(new_output[:, :192], new_output[:, 192:])

        return class_out
