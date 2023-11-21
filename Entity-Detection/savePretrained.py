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

model_id = "dmis-lab/biobert-v1.1"
output_model_path = "BC5CDR_model/pretrained"
output_tokenizer_path = "BC5CDR_model/pretrained"
loss_fn = clf_distill_loss_functions.Plain()


model = MyModel.from_pretrained(
        model_id,loss_fn = loss_fn
    )
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_tokenizer_path)