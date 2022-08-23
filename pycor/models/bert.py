from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pycor.load_annotations.load_with_bert import Sense_Selection_Data, SentDataset, collate_batch


# BERT_MODEL = 'Maltehb/danish-bert-botxo'
# MODEL, TOKENIZER, FORWARD = get_model_and_tokenizer('Maltehb/danish-bert-botxo',
#                                                    'bertbase',# 'bert_token_cos'
#                                                     device,
#                                                     checkpoint=r'pycor\models\checkpoints\model_0.pt')
# '/content/drive/MyDrive/SPECIALE/data/model_0.pt')


class BERT_model(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.device_attr = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BertModel(config)

        self.sigmoid = torch.nn.Sigmoid()
        self.post_init()

    def load_tokenizer(self, name):
        tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=True)

        # add new special token
        if '[TGT]' not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
            assert '[TGT]' in tokenizer.additional_special_tokens
            self.resize_token_embeddings(len(tokenizer))

        self.tokenizer = tokenizer
        return tokenizer

    def save_checkpoint(self, path, valid_loss):
        state_dict = {'model_state_dict': self.state_dict(),
                      'valid_loss': valid_loss}

        torch.save(state_dict, path)
        print(f'Saved model to: {path}')

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        print(f'Loaded model from: {path}')
        print(state_dict.keys())
        self.load_state_dict(state_dict['model_state_dict'])
        return state_dict['valid_loss']

    def save_metrics(self, path, train_loss_list, valid_loss_list, global_steps_list):
        state_dict = {'train_loss_list': train_loss_list,
                      'valid_loss_list': valid_loss_list,
                      'global_steps_list': global_steps_list}

        torch.save(state_dict, path)
        print(f'Saved model to: {path}')

    def load_metrics(self, path):
        state_dict = torch.load(path, map_location=self.device)
        print(f'Loaded model from: {path}')

        return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

    def get_BERT_score(self, data, output_type='data'):
        reduction = SentDataset(Sense_Selection_Data(data, self.tokenizer))
        dataloader = DataLoader(reduction,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=collate_batch)

        nb_eval_steps = 0
        score = []

        #iterator = tqdm(dataloader, desc="Iteration")

        for batches in dataloader:#iterator:
            if len(batches) < 1:
                continue

            self.eval()
            with torch.no_grad():
                # run model
                for batch in batches:
                    logits = self(input_ids=batch[2].to(self.device),
                                  attention_mask=batch[3].to(self.device),
                                  token_type_ids=batch[4].to(self.device)
                                  )

                    logits = self.sigmoid(torch.arctan(logits))
                    score.append(logits.cpu())

            nb_eval_steps += 1

        if output_type == 'data':
            data['score'] = torch.tensor(score)
            return data
        else:
            return torch.tensor(score)

    def get_bert_embedding(self, row):
        bert_data = SentDataset(Sense_Selection_Data(row, self.tokenizer, data_type='single'))

        self.eval()
        with torch.no_grad():
            for batch in bert_data:
                bert_out = self.bert(input_ids=torch.tensor(batch.input_ids).to(self.device).unsqueeze(0),
                                     attention_mask=torch.tensor(batch.input_mask).to(self.device).unsqueeze(0),
                                     token_type_ids=torch.tensor(batch.segment_ids).to(self.device).unsqueeze(0)
                                    )

                embedding = bert_out[1].squeeze().detach().cpu().numpy()

        return embedding


class BertSense(BERT_model):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Tanh() #
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(self.config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None):
        # batch = tuple(tensor.to(self.device) for tensor in batch[2:])
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
        # returns the last hidden layer of the classification token further processed by a Linear layer
        # and a Tanh activation function
        bert_out = self.dropout(bert_out[1])
        # linear = model.relu(model.linear(bert_out))
        # class_out = model.out(linear)
        class_out = self.out(bert_out)

        return class_out.squeeze(-1)


class BertSenseToken(BERT_model):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.activation = torch.nn.Tanh()
        self.linear = torch.nn.Linear(config.hidden_size * 2, 192)
        self.out = torch.nn.Linear(192, 1)
        self.post_init()

    @staticmethod
    def get_token_repr_idx(input_ids: torch.tensor):
        batch_size = input_ids.shape[0]
        placement = input_ids == 31748
        token_idxs = placement.nonzero().transpose(1, 0)
        return [token_idxs[1][token_idxs[0] == b] for b in range(batch_size)]

    @staticmethod
    def get_repr_avg(output_hidden_states, token_idx):
        layers_hidden = output_hidden_states[4:-4]

        if isinstance(layers_hidden, tuple):
            # layers_hidden = output_hidden_states[4:-4]
            layers_hidden = torch.mean(torch.stack(layers_hidden), axis=0)

        batch_size = layers_hidden.shape[0]
        hidden_token = [layers_hidden[b, token_idx[b][i] + 1:token_idx[b][j], :]
                        for b in range(batch_size) for i, j in [(0, 1), (-2, -1)]
                        ]

        hidden_token = torch.stack([torch.mean(hidden, dim=0)
                                    if hidden.shape[0] > 1 else hidden.squeeze(0)
                                    for hidden in hidden_token]).reshape(batch_size, -1, 768)

        if hidden_token.shape[0] > 4:
            hidden_token_1 = hidden_token[:2]
            hidden_token_2 = hidden_token[-2:]
            hidden_token = torch.concat((hidden_token_1, hidden_token_2), dim=1)

        # return hidden_token.reshape(batch_size, 2, 768)
        return hidden_token.reshape(batch_size, 2 * 768)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # batch = tuple(tensor.to(self.device) for tensor in batch[2:])
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

        hidden_states = bert_out.hidden_states

        token_ids = self.get_token_repr_idx(input_ids)
        new_output = self.get_repr_avg(hidden_states, token_ids)

        # returns the last hidden layer of the classification token further processed by a Linear layer
        # and a Tanh activation function
        # bert_out = model.dropout(bert_out[1])
        bert_out = self.dropout(new_output)
        linear = self.linear(self.activation(bert_out))
        # class_out = model.out(linear)
        class_out = self.out(linear)

        return class_out.squeeze(-1)

    def forward_cos(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                    inputs_embeds=None, output_attentions=None, return_dict=None):
        # batch = tuple(tensor.to(self.device) for tensor in batch[2:])
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

        bert_out = self.dropout(bert_out)
        hidden_states = bert_out.hidden_states
        token_ids = self.get_token_repr_idx(input_ids)
        new_output = self.get_repr_avg(hidden_states, token_ids)
        class_out = self.cos(new_output[:, :768], new_output[:, 768:])

        return class_out
