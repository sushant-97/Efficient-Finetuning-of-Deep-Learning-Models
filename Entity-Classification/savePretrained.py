import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from seqeval.metrics import classification_report
from config import Config as config
import os
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
# import clf_distill_loss_functions
# from clf_distill_loss_functions import *
import warnings

# from train_vanilla import MainModel
num_labels = 2  #128
model_id = "dmis-lab/biobert-v1.1"
output_model_path = "BC5CDR_MODEL_vanilla/pretrained"
output_tokenizer_path = "BC5CDR_MODEL_vanilla/pretrained"
# loss_fn = clf_distill_loss_functions.Plain()

class LossFunction(nn.Module):
    def forward(self, probability):
        loss = torch.log(probability)
        loss = -1 * loss
        # print(loss)
        loss = loss.mean()
        # print(loss)
        return loss

loss_fn = LossFunction()
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

special_tokens_dict = {'additional_special_tokens':['[ENTITY_START]','[ENTITY_END]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')


class MainModel(BertPreTrainedModel):
    def __init__(self, config, loss_fn = None):
        super(MainModel,self).__init__(config)
        # self.num_labels = num_labels
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained(model_id,config = config)
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        self.bert.resize_token_embeddings(28998)
        self.classifier = nn.Linear(768,self.num_labels)

    def forward(self, input_ids, attention_mask, labels,device):
              
        output = self.bert(input_ids, attention_mask = attention_mask)
        output = output.last_hidden_state
        output = output[:,0,:]
        classifier_out = self.classifier(output)
        main_prob = F.softmax(classifier_out, dim = 1)
        main_gold_prob = torch.gather(main_prob, 1, labels)
        loss_main = self.loss_fn.forward(main_gold_prob)
        return loss_main,main_prob

# config = AutoConfig.from_pretrained(model_id , num_labels=num_labels)
# model = MainModel.from_pretrained(model_id, config=config, loss_fn = loss_fn)


model = MainModel.from_pretrained(
        model_id, num_labels = 2, loss_fn = loss_fn, ignore_mismatched_sizes=True
    )
# config = "sushant/Entity-classification_/BC5CDR_MODEL_vanilla/pretrained/config.json"
# # Get the vocabulary size from the configuration
# vocab_size = config.vocab_size
# print('*******************************')
# print(f"Model vocabulary size: {vocab_size}")

model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_tokenizer_path)