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
import clf_distill_loss_functions
from clf_distill_loss_functions import *
import warnings

from train import MyModel

pretrained = "BC5CDR_model/pretrained"
class LossFunction(nn.Module):
    def forward(self, probability):
        loss = torch.log(probability)
        loss = -1 * loss
        # print(loss)
        loss = loss.mean()
        # print(loss)
        return loss

# loss_fn = clf_distill_loss_functions.Plain()
loss_fn = LossFunction()
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = MyModel.from_pretrained(pretrained, num_labels = 2,loss_fn = loss_fn)
device = 'cuda' if cuda.is_available() else 'cpu'
#loading model to device
model.to(device)

# sent = "I am feeling lucky"
# tk = tokenizer(sent)
# print(tk)
# print(True)