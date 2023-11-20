
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
from data_loader import load_data

# Ignore all warnings
warnings.filterwarnings("ignore")


input_path = './'
output_path = 'resources'
log_soft = F.log_softmax
tokenizer_dir = "./tokenizer"
model_dir = "./model"
config_dir = "./config"
print(torch.version.cuda)
MAX_LEN = 310# suitable for all datasets
BATCH_SIZE = 24
LEARNING_RATE = 1e-5
num_labels = 2      # 128 ## CHANGE
pretrained = "BC5CDR_MODEL_vanilla/pretrained"

class LossFunction(nn.Module):
    def forward(self, probability):
        loss = torch.log(probability)
        loss = -1 * loss
        # print(loss)
        loss = loss.mean()
        # print(loss)
        return loss

class MainModel(BertPreTrainedModel):
    def __init__(self, config, loss_fn = None):
        super(MainModel,self).__init__(config)
        # self.num_labels = num_labels
        self.num_labels = num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained(pretrained,config = config)
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
        
def train(model, dataloader, optimizer, device):
    tr_loss, tr_accuracy = 0, 0
    bias_loss = 0
    # tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    #put model in training mode
    model.train()
    
    for idx, batch in enumerate(dataloader):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, labels=targets,device = device)

        # print(f'\tLoss Main : {loss_main}')
        tr_loss += loss_main.item()
        nb_tr_steps += 1
        #compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
        if idx % 100 == 0:
            print(f'\tModel loss at {idx} steps: {tr_loss}')
            if idx != 0:
                print(f'\tModel Accuracy : {tr_accuracy/nb_tr_steps}')
            with open('live.txt', 'a') as fh:
                fh.write(f'\tModel Loss at {idx} steps : {tr_loss}\n')
                if idx != 0:
                    fh.write(f'\tModel Accuracy : {tr_accuracy/nb_tr_steps}')
        # print(f'Main loss : {loss_main} Bias Loss : {loss_bias} Accuracy : {tmp_tr_accuracy}')
        # print(tmp_tr_accuracy)
        # print(model.bert.encoder.layer[0].state_dict())
        # print("                    2nd Output                  ")
        torch.nn.utils.clip_grad_norm_(
            parameters = model.parameters(),
            max_norm = 10
        )
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()
        

    print(f'\tModel loss for the epoch: {tr_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy/nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy : {tr_accuracy/nb_tr_steps}\n')


def valid(model, dataloader, device):
    eval_loss = 0
    bias_loss = 0
    eval_accuracy = 0
    model.eval()
    nb_eval_steps = 0
    for batch in dataloader:
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, labels=targets,device = device)
        eval_loss += loss_main.item()
        nb_eval_steps += 1
        #compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy
        
    print(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tValidation Loss : {eval_loss}\n')
        fh.write(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}\n')
    return eval_loss, eval_accuracy/nb_eval_steps 



def main():
    print("Training model :")
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_model_directory', type=str, required=True)
    parser.add_argument('--output_tokenizer_directory', type=str, required=True)
    
    args = parser.parse_args()
    subset = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # subset = [90]

    for fr in subset:
        output_model_path = os.path.join(input_path, args.output_model_directory, str(fr))
        output_tokenizer_path = os.path.join(input_path, args.output_tokenizer_directory, str(fr))
        

        with open('live.txt', 'a') as fh:
            fh.write(f'Subset : {fr}\n')
            fh.write(f'Dataset : {args.dataset_name}\n')
            fh.write(f'Model Path : {output_model_path}\n')
            
            
        best_output_model_path = output_model_path   
        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)
        if not os.path.exists(best_output_model_path):
            os.makedirs(best_output_model_path)

        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        special_tokens_dict = {'additional_special_tokens':['[ENTITY_START]','[ENTITY_END]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print('We have added', num_added_toks, 'tokens')

        id2label, label2id, train_data = load_data(args.dataset_name, 'train', tokenizer, fr)
        # print(train_data[0])
        _,_,devel_data = load_data(args.dataset_name, 'devel', tokenizer, fr)
        # config = AutoConfig.from_pretrained(pretrained , num_labels=num_labels)
        model = MainModel.from_pretrained(pretrained, num_labels=num_labels, loss_fn = LossFunction())

        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
        device = 'cuda' if cuda.is_available() else 'cpu'
        model.to(device)
        train_dataloader = DataLoader(train_data, shuffle = True, batch_size=BATCH_SIZE)
        devel_dataloader = DataLoader(devel_data, shuffle=True, batch_size=BATCH_SIZE)
        num_epochs = 20
        max_acc = 0.0
        patience = 0
        best_model = model
        best_tokenizer = tokenizer
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}:')
            with open('live.txt', 'a') as fh:
                fh.write(f'Epoch : {epoch+1}\n')
            train(model, train_dataloader, optimizer, device)
            validation_loss, eval_acc = valid(model, devel_dataloader, device)
            print(f'\tValidation loss: {validation_loss}')
            if round(eval_acc,4) >= round(max_acc, 4):
                max_acc = round(eval_acc, 4)
                patience = 0
                best_model = model
                best_tokenizer = tokenizer
            else:
                patience += 1
                if patience > 3:
                    print("Early stopping at epoch : ",epoch)
                    best_model.save_pretrained(best_output_model_path)
                    best_tokenizer.save_pretrained(best_output_model_path)
                    patience = 0
                    break

        # model.save_pretrained(output_model_path)
        # tokenizer.save_pretrained(output_tokenizer_path)
        
        best_model.save_pretrained(best_output_model_path)
        best_tokenizer.save_pretrained(best_output_model_path)

        end = time.time()
        total_time = end - start
        with open('live.txt', 'a') as fh:
            fh.write(f'Total training time : {total_time}\n')

        print(f"Total training time : {total_time}")

if __name__ == '__main__':
    main()