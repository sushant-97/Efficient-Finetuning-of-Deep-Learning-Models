import os 
import time
import math 
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification,BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn.functional as F
from seqeval.metrics import classification_report
import torch.nn as nn
import clf_distill_loss_functions
from clf_distill_loss_functions import *
from clf_distill_loss_functions import ClfDistillLossFunction
from train import MyModel
input_path = './'
output_path = 'resources'
pretrained = "BC5CDR_model/pretrained"

MAX_LEN = 310
BATCH_SIZE = 16
count_labels = []

def read_data(dataset_name, test_file_name):
    train_path = os.path.join(input_path, test_file_name)
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

    test_path = os.path.join(input_path, test_file_name)
    test_token_lst, test_label_lst = [], []
    with open(test_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                test_token_lst.append(math.nan)
                test_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            test_token_lst.append(a[0].strip())
            test_label_lst.append(a[1].strip())

    test_data = pd.DataFrame({'Tokens': test_token_lst, 'Labels': test_label_lst})

    return train_data, test_data

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

    return sent_list,label_list
'''       
class MyModel(BertPreTrainedModel):
    def __init__(self, config,loss_fn: ClfDistillLossFunction):
        super(MyModel,self).__init__(config)
        self.num_labels = 3
        self.loss_fn = loss_fn
        self.bert = AutoModel.from_pretrained('./model')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)
        #self.init_weights()
        self.crf = CRF(3, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None,bias = None, teacher_probs = None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        attention_mask[:, 0] = 1
        attention_mask = attention_mask.type(torch.uint8)
        #logits = self.crf.decode(logits, mask=attention_mask)
        #logits = torch.tensor(logits)
        
        if bias is not None:
            attention_mask = attention_mask.type(torch.uint8)
            loss = self.loss_fn.forward(sequence_output, logits, bias, teacher_probs, labels)
            #return loss,logits
        else:
            lossFunction = nn.CrossEntropyLoss()
            loss = lossFunction(logits.view(-1,self.num_labels), labels.view(-1))
        
        return loss,logits
'''
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
        labels.extend([label]*n_subwords)

    return tokenized_sentence, labels 

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label2id, id2label):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer 
        self.max_len = max_len 
        self.label2id = label2id
        self.id2label = id2label

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
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len



def inference(model, dataloader, tokenizer, device, id2label):
    model.eval()
    pred_lst = []
    test_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols = 100)):
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        # outputs = model(model.num_labels, ids, mask, labels=targets)

        loss, inference_logits = outputs[0], outputs[1]
        test_loss += loss.item()
        nb_test_steps += 1
        if_logits = inference_logits
        #if_logits = F.softmax(inference_logits, dim=2)
        flattened_targets = targets.view(-1)
        flattened_predictions = if_logits.view(-1)
        #flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        test_accuracy += tmp_test_accuracy

        #inference_logits = F.softmax(inference_logits, dim=2)
        #inference_ids = torch.argmax(inference_logits, dim=2)
        for i in range(ids.shape[0]):
            tmp_labels = []
            tmp_test_tokens = tokenizer.convert_ids_to_tokens(ids[i])

            tmp_label_ids = inference_logits[i]
            for index, tok in enumerate(tmp_test_tokens):
                if tok in ['[CLS]', '[SEP]', '[PAD]']:
                    continue 
                else:
                    tmp_labels.append(id2label[tmp_label_ids[index].item()])
            
            pred_lst.append(tmp_labels)

    test_accuracy = test_accuracy / nb_test_steps
    print(f'\t Test accuracy: {test_accuracy}')
    return pred_lst


def generate_prediction_file(pred_labels, tokens, dataset_name, tokenizer, output_file):
    #p_name = 'preds_' + model_name + '.txt'
    #output_file = os.path.join(dataset_name, p_name)
    labels = pred_labels[0]
    i = 1
    j = 0
    with open(output_file, 'w') as fh:
        for tok in tokens:
            if isinstance(tok, float):
                fh.write('\n')
                if i >= len(pred_labels):
                    break
                labels = pred_labels[i]
                i += 1
                j = 0
            elif j < len(labels):
                sub_words = tokenizer.tokenize(tok)
                fh.write(f'{tok}\t{labels[j]}\n')
                j += len(sub_words)
            else:
                fh.write(f'{tok}\tO\n')
                

def main():
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--test_file_name', type=str, required=True)
    parser.add_argument('--model_directory', type=str, required=False)
    parser.add_argument('--tokenizer_directory', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--sample_fraction', type = str, required=False)
    
    args = parser.parse_args()
    
    # hf_model_id = "dmis-lab/biobert-v1.1"

    input_model_path = os.path.join(input_path, args.model_directory)
    input_tokenizer_path = os.path.join(input_path, args.tokenizer_directory)
    # input_model_path = hf_model_id
    # input_tokenizer_path = hf_model_id
    
    # read data
    train_data, test_data = read_data(args.dataset_name, args.test_file_name)
    # get a dict for label and its id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    #get list of sentence and associated label
    test_sent, test_label = convert_to_sentence(test_data)
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    # loading tokenizer
    loss_fn = clf_distill_loss_functions.Plain()
    tokenizer = AutoTokenizer.from_pretrained(
        input_tokenizer_path
    )
    model = MyModel.from_pretrained(
        input_model_path,loss_fn = loss_fn
    )

    num_labels = len(id2label)
    test_df = {'Sentence':test_sent, 'Labels':test_label}
    test_df = pd.DataFrame(test_df)

    test_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False,
                    }

    test_dataset = dataset(test_df, tokenizer, MAX_LEN, label2id, id2label)
    test_dataloader = DataLoader(test_dataset, **test_params)

  
    model.to(device)

    # getting predictions
    print("Testing started")
    pred_labels = inference(model, test_dataloader, tokenizer, device, id2label)
    # writing to file
    generate_prediction_file(pred_labels, test_data['Tokens'].tolist(), args.dataset_name, tokenizer, args.output_file)

    end = time.time()
    total_time = end - start
    print(f"Testing time : {total_time}")
    
if __name__ == '__main__':
    main()
