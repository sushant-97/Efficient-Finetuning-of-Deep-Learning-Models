from multiprocessing import reduction
import pandas as pd
import time
import numpy as np
import csv
import argparse
import math
from sklearn.metrics import accuracy_score
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
import warnings
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import random

input_path = './'
MAX_LEN = 310# suitable for all datasets


## Takes train.txt file with BIO tags and converts them into list of sentences and lables
def read_data(dataset_name, data, fraction):
    if data == "train":
        data = data + '_' + str(fraction) + '.txt'
    else:
        data = data + '.txt'
        parts = dataset_name.split('/')
        new_parts = [part for part in parts if part != 'R1' and part != 'R2' and part != 'R3']
        dataset_name = os.path.join(*new_parts)
        print(dataset_name)
    path = os.path.join(input_path, dataset_name, data)
    token_lst, label_lst = [], []
    sentences = []
    labels = []
    all_labels = []
    with open(path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                sentences.append(token_lst)
                labels.append(label_lst)
                token_lst = []
                label_lst = []
                continue
            a = line.split('\t')
            token_lst.append(a[0].strip())
            label_lst.append(a[1].strip())
    label_path = 'labels.txt'
    parts = dataset_name.split('/')
    new_parts = [part for part in parts if part != 'R1' and part != 'R2' and part != 'R3']
    new_path = os.path.join(*new_parts)

    label_path = os.path.join(input_path, new_path, label_path)
    with open(label_path, 'r') as fh:
        for line in fh:
            if(len(line.strip()) != 0):
                all_labels.append(line.strip())
    if(dataset_name == 'MedMentions'):
        assert(len(all_labels) == 128)
    return sentences, labels, all_labels


## Adding same label to the tokenized word.. as word may be split into multiple tokens
def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        if(len(tokenized_word) == 0 and len(word) != 0):
            tokenized_word = ' '
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        # add same label of word to other subwords
        if label[0] != 'O':
            labels.extend([label])
            new_label = 'I' + label[1:]
            labels.extend([new_label] * (n_subwords-1))
        else:
            labels.extend(label * (n_subwords))

    return tokenized_sentence, labels 


def IdToLabelAndLabeltoId(label_list):
    label_list = [x for x in label_list if not pd.isna(x)]
    # sorting as applying set operation does not maintain the order
    label_list.sort()
    # print(label_list)
    id2label = {}
    for index, label in enumerate(label_list):
        id2label[index] = label
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id


class dataset(Dataset):
    def __init__(self, sentence_list, labels_list, tokenizer, max_len, label2id, id2label):
        self.len = len(sentence_list)
        self.sentence = sentence_list
        self.labels = labels_list
        self.tokenizer = tokenizer
        self.max_len = max_len 
        self.label2id = label2id
        self.id2label = id2label
        self.maximum_across_all = 0 

    def __getitem__(self, index):
        # step 1: tokenize sentence and adapt labels
        sentence = self.sentence[index]
        label = self.labels[index]
        label2id = self.label2id
        tokenized_sentence = ['[CLS]'] + sentence + ['[SEP]']
        
        # step 3: truncating or padding
        max_len = self.max_len
        #print(tokenized_sentence)
        if len(tokenized_sentence) > max_len:
            #truncate
            tokenized_sentence = tokenized_sentence[:max_len]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(max_len - len(tokenized_sentence))]

        # step 4: obtain attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        # print(tokenized_sentence)
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_id = label2id[label]
        target = []
        target.append(label_id)

        return {
            'index': index,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

    def __len__(self):
        return self.len




def load_data(dataset_name, data, tokenizer, fraction):
    sentences,labels, all_labels = read_data(dataset_name, data, fraction)
    
    sentence_list = []
    labels_list = []
    for sentence,label in zip(sentences,labels):
        entity_list = []
        tokenized_sentence,tokenized_label = tokenize_and_preserve_labels(sentence, label, tokenizer)
        for i in range(len(tokenized_label)):
            if(tokenized_label[i][0] == 'B'):
                l = i
                entity_class = tokenized_label[i][1:]   # 1 for BC5CDR and 2 for MedMentions
                # print(entity_class)
                i += 1
                while(i < len(tokenized_label) and tokenized_label[i][0] == 'I'):
                    i += 1
                r = i
                i -= 1
                entity_list.append([l,r,entity_class])
        for entity in entity_list:
            new_sent = []
            for i in range(len(tokenized_label)):
                if(i == entity[0]):
                    new_sent.append('[ENTITY_START]')
                elif(i == entity[1]):
                    new_sent.append('[ENTITY_END]')
                new_sent.append(tokenized_sentence[i])
            if(entity[1] == len(tokenized_label)):
                new_sent.append('[ENTITY_END]')
            sentence_list.append(new_sent)
            labels_list.append(entity[2])
    print("No of datapoints i.e. Entities to classify")
    print(len(sentence_list))
    print(len(labels_list))
    
    # #Save gold labels
    # path = "./"
    # output_file = os.path.join(path, dataset_name, "gold_label.txt")
    # print(output_file)
    # with open(output_file, 'a') as fh:
    #     for lab in labels_list:
    #         fh.write(f'{lab}\n')
    
    id2label,label2id = IdToLabelAndLabeltoId(all_labels)
    data  = dataset(sentence_list=sentence_list, labels_list=labels_list, tokenizer=tokenizer,\
                          max_len=MAX_LEN, id2label=id2label, label2id=label2id)
    return id2label, label2id, data

