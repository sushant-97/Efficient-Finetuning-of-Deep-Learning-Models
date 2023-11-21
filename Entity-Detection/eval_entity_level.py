import argparse
import time
import os
import re
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from collections import Counter
from tqdm import tqdm
from string import punctuation
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 

input_path = './'

def read_data(dataset_name, data_file):
    path = os.path.join(input_path, data_file)
    token_lst, label_lst = [], []
    sentences = []
    labels = []
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
    return sentences, labels

def find_num_overlap(entity_list, pred_label):
    num_overlap = 0
    for entity in entity_list:
        for i in range(entity[0],entity[1]):
            if(pred_label[i] == 'B' or pred_label[i] == 'I'):
                num_overlap += 1
                break
    return num_overlap


def eval(sentences_gold, labels_gold, sentences_pred, labels_pred):
    true_positive = 0
    false_negative = 0
    for i in range(len(sentences_gold)):
        gold_sentence = sentences_gold[i]
        gold_label = labels_gold[i]
        pred_sentence = sentences_pred[i]
        pred_label = labels_pred[i]
        entity_list = []
        for i in range(len(gold_label)):
            if(gold_label[i][0] == 'B'):
                l = i
                i += 1
                while(i < len(gold_label) and gold_label[i][0] == 'I'):
                    i += 1
                r = i
                i -= 1
                entity_list.append([l,r])
        num_entities = len(entity_list)
        num_overlap = find_num_overlap(entity_list, pred_label)
        true_positive += num_overlap
        false_negative += (num_entities - num_overlap)
    
    return true_positive, false_negative



def main():
    # arguments
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--sample_fraction', type=str, required=False)
    parser.add_argument('--iteration', type=str, required=False)
    args = parser.parse_args()
    print(args.gold_file)
    print(args.pred_file)
    sentences_gold, labels_gold = read_data(args.dataset_name, args.gold_file)
    sentences_pred, labels_pred = read_data(args.dataset_name, args.pred_file)
    assert(len(sentences_gold) == len(sentences_pred))

    true_positive, false_negative = eval(sentences_gold, labels_gold, sentences_pred, labels_pred)
    _, false_positive = eval(sentences_pred, labels_pred, sentences_gold, labels_gold)
    precision = true_positive/(true_positive + false_positive)
    precision *= 100
    recall = true_positive/(true_positive + false_negative)
    recall *= 100
    f1_score = (2 * precision * recall)/(precision + recall)
    print(f'Precision : {precision}\nRecall : {recall}\nF1-Score : {f1_score}')
    
    with open(f'eval_BC5CDR_IFR.txt', 'a') as fh:
        fh.write(f'Results for {args.sample_fraction}\n')
        fh.write(f'Precision : {precision}\nRecall : {recall}\nF1-Score : {f1_score}\n')
        fh.write('\n\n')    

    end = time.time()

    total_time = end - start

    print(f"Evaluation time : {total_time}\n\n")

if __name__ == '__main__':
    main()