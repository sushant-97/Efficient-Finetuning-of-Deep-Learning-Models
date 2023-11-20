
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
import clf_distill_loss_functions
from clf_distill_loss_functions import *
import warnings
from sklearn.model_selection import train_test_split

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
MAX_GRAD_NORM = 10
BATCH_SIZE = 64
LEARNING_RATE = 3e-5
pretrained = "MED_model/pretrained"

def read_data(dataset_name, fraction):
    
    train_path = os.path.join(input_path, dataset_name, f'C/train_{fraction}.txt')
    
    devel_path = os.path.join(input_path, dataset_name, 'devel.txt')
    train_token_lst, train_label_lst = [], []
    with open(train_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                train_token_lst.append(math.nan)
                train_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            train_token_lst.append(a[0].strip())
            train_label_lst.append(a[1].strip())

    train_data = pd.DataFrame({'Tokens': train_token_lst, 'Labels': train_label_lst})

    devel_token_lst, devel_label_lst = [], []
    with open(devel_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                devel_token_lst.append(math.nan)
                devel_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            devel_token_lst.append(a[0].strip())
            devel_label_lst.append(a[1].strip())

    devel_data = pd.DataFrame({'Tokens': devel_token_lst, 'Labels': devel_label_lst})

    return train_data, devel_data

def IdToLabelAndLabeltoId(train_data):
    label_list = train_data["Labels"]
    label_list = [*set(label_list)]
    label_list = [x for x in label_list if not pd.isna(x)]
    # sorting as applying set operation does not maintain the order
    label_list.sort()
    id2label = {}
    for index, label in enumerate(label_list):
        id2label[index] = label
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id

def convert_to_sentence(df):
    sent = ""
    sent_list = []
    label = ""
    label_list = []
    for tok,lab in df.itertuples(index = False):
        if isinstance(tok, float):
            sent = sent[1:]
            sent_list.append(sent)
            sent = ""
            label = label[1:]
            label_list.append(label)
            label = ""
        else:
            sent = sent + " " +str(tok)
            label = label+ "," + str(lab)
    if sent != "":
        sent_list.append(sent)
        label_list.append(label)

    # # Prince yadav -- writing logic to select x % of the data
    # testsize = 0.00
    # #X_train, X_test, y_train, y_test = train_test_split(sent_list, label_list, test_size=testsize)
    # print(1-testsize)
    # X_train = sent_list
    # y_train = label_list
    # return X_train, y_train
    return sent_list,label_list

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
    sentence = str(sentence).strip()
    text_labels = str(text_labels)

    for word, label in zip(sentence.split(), text_labels.split(',')):
        # tokenize and count num of subwords
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        # add same label of word to other subwords
        if label != 'O':
            labels.extend([label])
            labels.extend(['I'] * (n_subwords-1))
        else:
            labels.extend(['O'] * (n_subwords))

    return tokenized_sentence, labels 

def loadBiasProb(dataset_name, bias_file):
    bias_file_path = os.path.join(bias_file)
    bias_probs = []
    probs_per_sent = []
    with open(bias_file_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                if len(probs_per_sent) > 0:
                    bias_probs.append(probs_per_sent)
                    probs_per_sent = []
            else:
                parts = line.split(',')
                tmp_probs = []
                max_pos, max_val = 0, float('-inf')
                to_subtract = 0
                for i, p in enumerate(parts):
                    tmp_probs.append(float(p))
                    if float(p) > max_val:
                        max_val = float(p)
                        max_pos = i
                    if float(p) == 0:
                        tmp_probs[i] = 0.001
                        to_subtract += 0.001

                
                tmp_probs[max_pos] -= to_subtract
                probs_per_sent.append(tmp_probs)
                #tensor_vals = torch.tensor(tmp_probs)
                #tensor_vals = F.softmax(tensor_vals,dim = 0)
                #tensor_vals = tensor_vals.tolist()
                
                #probs_per_sent.append(tensor_vals)

    return bias_probs

def LoadTeacherProb(teacher_file):
    main_probs, probs_per_sent = [],[]
    main_file_path = os.path.join(teacher_file)
    with open(main_file_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                if len(probs_per_sent) > 0:
                    main_probs.append(probs_per_sent)
                    probs_per_sent = []
            else:
                parts = line.split(',')
                tmp_probs = []
                max_pos, max_val = 0, float('-inf')
                to_subtract = 0
                for i, p in enumerate(parts):
                    tmp_probs.append(float(p))
                    if float(p) > max_val:
                        max_val = float(p)
                        max_pos = i
                    if float(p) == 0:
                        tmp_probs[i] = 0.001
                        to_subtract += 0.001

                tmp_probs[max_pos] -= to_subtract
                probs_per_sent.append(tmp_probs)
    return main_probs
    
    
    
class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label2id, id2label):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer 
        self.max_len = max_len 
        self.label2id = label2id
        self.id2label = id2label
        self.maximum_across_all = 0 

    def __getitem__(self, index):
        # step 1: tokenize sentence and adapt labels
        sentence = self.data.Sentence[index]
        word_labels = self.data.Labels[index]
        label2id = self.label2id

        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)

        # step 2: add special tokens and corresponding labels
        tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
        labels.insert(0, 'O')
        labels.insert(-1, 'O')
        
        # step 3: truncating or padding
        max_len = self.max_len

        if len(tokenized_sentence) > max_len:
            #truncate
            tokenized_sentence = tokenized_sentence[:max_len]
            labels = labels[:max_len]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(max_len - len(tokenized_sentence))]
            labels = labels + ['O' for _ in range(max_len - len(labels))]

        # step 4: obtain attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [label2id[label] for label in labels]

        return {
            'index': index,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len
      
        
class MyModel(BertPreTrainedModel):
    def __init__(self, config,loss_fn: ClfDistillLossFunction):
        super(MyModel,self).__init__(config)
        self.num_labels = 3
        self.loss_fn = loss_fn
        self.bert = AutoModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)
        #self.init_weights()
        self.crf = CRF(3, batch_first=True)

    def forward(self,input_ids, attention_mask, labels=None,bias = None, teacher_probs = None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # print("hereher ************************")
        # print(logits.shape)
        # print("hereher ************************")
        if bias is not None:
            attention_mask = attention_mask.type(torch.uint8)
            loss = self.loss_fn.forward(self.num_labels,sequence_output, logits, bias, teacher_probs, labels)
            #return loss,logits
        else:
            lossFunction = nn.CrossEntropyLoss()
            loss = lossFunction(logits.view(-1,self.num_labels), labels.view(-1))
        logits = self.crf.decode(logits)
        logits = torch.tensor(logits).to('cuda')
        return loss,logits

def train(model, dataloader, optimizer, device, sent_lst, tokenizer):
    tr_loss, tr_accuracy = 0, 0
    # tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    #put model in training mode
    model.train()
    
    for idx, batch in enumerate(dataloader):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.long)

        # bias_prob,teacher_prob = [],[]
        # b_no = 0
        
        # for index in indexes:
        #     sentence = sent_lst[index]
        #     words = sentence.split(' ')
        #     # bias_prob_per_sent = []
        #     # teacher_prob_per_sent = []
        #     # bias_prob_per_sent.append([0.001, 0.001, 0.998]) # for [CLS] token
        #     # teacher_prob_per_sent.append([0.001, 0.001, 0.998])
        #     for i, word in enumerate(words):
        #         subwords = tokenizer.tokenize(word)
        #         for _ in range(len(subwords)):
        #             if i < len(teacher_probability[index]):
        #                 teacher_prob_per_sent.append(teacher_probability[index][i])
        #                 bias_prob_per_sent.append(bias_probability[index][i])
        #             else:
        #                 teacher_prob_per_sent.append([0.001, 0.001, 0.998])
        #                 bias_prob_per_sent.append([0.001, 0.001, 0.998])
        #             #bias_prob_per_sent.append(bias_probability[index][i])
        #             #teacher_prob_per_sent.append(teacher_probability[index][i])
        #     bias_prob_per_sent.append([0.001, 0.001, 0.998]) # for [SEP] token
        #     teacher_prob_per_sent.append([0.001, 0.001, 0.998]) # for [SEP] token
        #     for m in mask[b_no]: # for [PAD] tokens
        #         if m == 0:
        #             bias_prob_per_sent.append([0.001, 0.001, 0.998])
        #             teacher_prob_per_sent.append([0.001, 0.001, 0.998])
        #     b_no += 1
        #     bias_prob.append(bias_prob_per_sent)
        #     teacher_prob.append(teacher_prob_per_sent)
        # print(len(bias_prob), len(bias_prob[0]), len(bias_prob[0][0]))
        
        # bias_prob = torch.tensor(bias_prob, dtype=torch.float32)
        # # print(bias_prob.shape)
        # bias_prob = bias_prob.to(device, dtype=torch.float32)
        # teacher_prob = torch.tensor(bias_prob, dtype=torch.float32)
        # # print(bias_prob.shape)
        # teacher_prob = bias_prob.to(device, dtype=torch.float32)
        

        # outputs = model(input_ids=input_ids, attention_mask=mask, labels=targets, bias = bias_prob, teacher_probs = teacher_prob)
        outputs = model(input_ids=input_ids, attention_mask=mask, labels=targets)

        loss = outputs[0]
        tr_logits = outputs[1]

        tr_loss += loss.item()
        #tr_logits = F.softmax(tr_logits, dim=2)
        nb_tr_steps += 1
        if idx % 100 == 0:
            print(f'\tTraining loss at {idx} steps: {tr_loss}')
            with open('live.txt', 'a') as fh:
                fh.write(f'\tTraining Loss at {idx} steps : {tr_loss}\n')

        #compute training accuracy
        flattened_targets = targets.view(-1)
        flattened_predictions = tr_logits.view(-1)
        #print(flattened_targets.shape)
        #print(flattened_predictions.shape)
        #flattened_predictions = torch.argmax(active_logits, dim=1)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        # tr_preds.extend(predictions)
        # tr_labels.extend(targets)

        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters = model.parameters(),
            max_norm = 10
        )

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'\tTraining loss for the epoch: {tr_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy/nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy : {tr_accuracy/nb_tr_steps}\n')


def valid(model, dataloader, device):
    eval_loss = 0
    eval_accuracy = 0
    model.eval()
    nb_eval_steps = 0
    for batch in dataloader:
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=input_ids, attention_mask=mask, labels=targets)

        loss = outputs[0]
        eval_logits = outputs[1]
        eval_loss += loss.item()
        nb_eval_steps += 1
        #compute training accuracy
        flattened_targets = targets.view(-1)
        flattened_predictions = eval_logits.view(-1)
        #print(flattened_targets.shape)
        #print(flattened_predictions.shape)
        #flattened_predictions = torch.argmax(active_logits, dim=1)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        # tr_preds.extend(predictions)
        # tr_labels.extend(targets)

        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
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
    # parser.add_argument('--bias_file', type=str, required=False)
    # parser.add_argument('--teacher_file', type=str, required=False)
    parser.add_argument('--loss_fn', type=str, required=True)
    parser.add_argument('--output_model_directory', type=str, required=True)
    parser.add_argument('--output_tokenizer_directory', type=str, required=True)
    parser.add_argument('--sample_fraction', type = str, required=False)
    
    args = parser.parse_args()
    
    output_model_path = os.path.join(input_path, args.output_model_directory)
    output_tokenizer_path = os.path.join(input_path, args.output_tokenizer_directory, f'{args.sample_fraction}')
    
    
    print("Dataset name : ", args.dataset_name)
    print('Model to be trained on ',args.loss_fn)
    print('Model will be saved on ', output_model_path)  
     
     
    with open('live.txt', 'a') as fh:
        fh.write(f'Dataset : {args.dataset_name}\n')
        fh.write(f'Loss : {args.loss_fn}\n')
        fh.write(f'Model Path : {output_model_path}\n')
    

    best_output_model_path = output_model_path + f'/{args.sample_fraction}'    
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    if not os.path.exists(best_output_model_path):
        os.makedirs(best_output_model_path)
        
    # read data
    train_data, devel_data = read_data(args.dataset_name, args.sample_fraction)
    
    # get a dict for label and its id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    num_labels = len(id2label)

    #get list of sentence and associated label
    train_sent, train_label = convert_to_sentence(train_data)
    devel_sent,devel_label = convert_to_sentence(devel_data)

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    tokenizer.save_pretrained("tokenizer")
    # setting the loss function
    if args.loss_fn == 'SmoothedDistillLoss':
        loss_fn = clf_distill_loss_functions.SmoothedDistillLoss()
    elif args.loss_fn == 'ReweightBaseline':
        loss_fn = clf_distill_loss_functions.ReweightBaseline()
    elif args.loss_fn == 'LearnedMixinBaseline':
        loss_fn = clf_distill_loss_functions.LearnedMixinBaseline(0.03)
    elif args.loss_fn == 'BiasProductBaseline':
        loss_fn = clf_distill_loss_functions.BiasProductBaseline()
    elif args.loss_fn == 'NewDistillLoss':
        loss_fn = clf_distill_loss_functions.NewDistillLoss()
    else:
        loss_fn = clf_distill_loss_functions.Plain()


    model = MyModel.from_pretrained(pretrained,num_labels = len(label2id),loss_fn = loss_fn)
    device = 'cuda' if cuda.is_available() else 'cpu'
    #loading model to device
    model.to(device)

    train_data = {'Sentence':train_sent, 'Labels':train_label}
    train_data = pd.DataFrame(train_data)
    devel_data = {'Sentence':devel_sent, 'Labels':devel_label}
    devel_data = pd.DataFrame(devel_data)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False,
                    }

    devel_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    }
    
    train_dataset = dataset(train_data, tokenizer, MAX_LEN, label2id, id2label)
    train_dataloader = DataLoader(train_dataset, **train_params)
    devel_dataset = dataset(devel_data, tokenizer, MAX_LEN, label2id, id2label)
    devel_dataloader = DataLoader(devel_dataset, **devel_params)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    num_epochs = 25 # no reason, IEEEAccess paper used this, so trying with this number
    #early_stopping = EarlyStopping(patience=3, verbose=True)
    max_acc = 0.0
    patience = 0
    best_model = model
    best_tokenizer = tokenizer
    with open('live.txt', 'a') as fh:
            fh.write(f'Results for Subset : {args.sample_fraction}\n')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        with open('live.txt', 'a') as fh:
            fh.write(f'Epoch : {epoch+1}\n')
        train(model, train_dataloader, optimizer, device, train_sent, tokenizer)
        validation_loss, eval_acc = valid(model, devel_dataloader, device)
        print(f'\tValidation loss: {validation_loss}')
        
        if round(eval_acc, 2) > round(max_acc, 2):
            max_acc = eval_acc
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
                #break
        
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
