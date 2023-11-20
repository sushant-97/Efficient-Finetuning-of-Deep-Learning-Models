from multiprocessing import reduction
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
import csv
import argparse
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

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
from data_loader import load_data
from train_vanilla import MainModel as Model_vanilla
from train_vanilla import LossFunction as LossFunction_vanilla
# from train import MainModel as Model_e2e
# from train import LossFunction as LossFunction_e2e

input_path = './'
output_path = 'resources'


MAX_LEN = 310
BATCH_SIZE = 24
num_labels = 2  #128 MedMentions

def inference(model, dataloader, tokenizer, device, model_type, id2label):
    model.eval()
    pred_lst = []
    test_loss = 0
    bias_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        with torch.no_grad():
            if model_type == 'vanilla':
                loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, labels=targets, device = device)
            else:
                loss_main,_, main_prob = model(input_ids=input_ids, attention_mask=mask, labels=targets, device = device)
        test_loss += loss_main.item()
        nb_test_steps += 1
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        labels = [id2label[class_id.item()] for class_id in predicted_labels]
        test_accuracy += tmp_test_accuracy
        pred_lst.extend(labels)
        
    test_accuracy = test_accuracy / nb_test_steps
    return test_accuracy, pred_lst


def generate_prediction_file(pred_lst, output_file, fr):
    output_file = output_file + "_" + str(fr) + ".txt"
    with open(output_file, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')


def eval(pred_label):

    label_list = []
    with open("BC5CDR/labels.txt", 'r') as fh:
        for line in fh:
            line = line.strip()
            label_list.append(line)
    
    gold_labels_path = "BC5CDR/gold_label.txt"
    gold_label = []
    with open(gold_labels_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            gold_label.append(line)
    print(len(gold_label), len(pred_label))
    precision_, recall_, f1_score_, _ = precision_recall_fscore_support(gold_label, pred_label,average="weighted")
    # precision_ = precision_score(gold_label, pred_label, average="weighted")
    # recall_ = recall_score(gold_label, pred_label, average="weighted")
    # f1_score_ = f1_score(gold_label, pred_label, average="weighted")

    return precision_, recall_, f1_score_

def main():
    
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_directory', type=str, required=True)
    parser.add_argument('--tokenizer_directory', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    
    args = parser.parse_args()
    with open('result.txt', 'a') as fh:
            fh.write(f'New Trial{args.dataset_name}\n')
    
    # subset = ["pretrained"]
    subset = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # subset = [90]
    for fr in subset:
        fr = str(fr)
        input_model_path = os.path.join(input_path, args.model_directory,fr)
        input_tokenizer_path = os.path.join(input_path, args.tokenizer_directory, fr)

        output_file_path = os.path.join(input_path, args.output_file)
        tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_path)
        id2label, label2id, test_data = load_data(args.dataset_name, 'test', tokenizer, fr)

        model = Model_vanilla.from_pretrained(input_model_path, num_labels = num_labels, loss_fn = LossFunction_vanilla())
        
        device = 'cuda' if cuda.is_available() else 'cpu'
        model.to(device)
        test_dataloader = DataLoader(test_data, shuffle = False, batch_size=BATCH_SIZE)
        test_accuracy, pred_lst = inference(model, test_dataloader, tokenizer, device, args.model_type, id2label)
        pred_lst_path = "MedMentions_PRED_vanilla/BP_test_preds_MedMentions_pretrained.txt"
        generate_prediction_file(pred_lst, output_file_path, fr)

        precision_, recall_, F1_score_ = eval(pred_lst)

        print("******************************************")
        print(str(fr))
        print(precision_, recall_, F1_score_)
        # print(f'\t{args.dataset_name} test accuracy: {test_accuracy}')

        #Store the results
        with open('result.txt', 'a') as fh:
            fh.write(f'Result for\t{str(fr)}:\n')
            fh.write(f'{args.dataset_name}\t test accuracy: {round(test_accuracy, 2)}\n')
            fh.write(f'Precision: {round(precision_*100, 4)}\n')
            fh.write(f'Recall: {round(recall_*100, 4)}\n')
            fh.write(f'F1_score: {round(F1_score_*100, 4)}\n')
            fh.write("\n")

        end = time.time()
        total_time = end - start
        with open('live.txt', 'a') as fh:
            fh.write(f'Total training time : {total_time}\n')

        print(f"Total training time : {total_time}")

if __name__ == '__main__':
    main()
