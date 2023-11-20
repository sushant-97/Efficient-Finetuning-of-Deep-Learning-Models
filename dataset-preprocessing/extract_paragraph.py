import spacy 
import os
import argparse
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
from train import MyModel
from sentence_transformers import SentenceTransformer

input_path = './'
output_path = 'resources'
pretrained = "BC5CDR_model/pretrained"

# 	python3 extract_paragraph.py --dataset BC5CDR/train_corpus.txt

MAX_LEN = 310
BATCH_SIZE = 16

#___________Approach 3 - Comparing text rather than position________

special_sym = ['-', '+', '(', ')', '[', ']']
# special_sym = ['-', '+', '(', ')']


def extract_paragraph(dataset, is_medmention=False, pmid_file_path=""):
    # delete output file if it exists (needed as append used while writing to file)
    # if os.path.exists(processed_file_path):
    #     os.remove(processed_file_path)
    #     print(f'{processed_file_path} was removed.')
    
    curr_pmid = []
    titles = []
    paragraphs = []
    
    print("herere")
    with open(dataset, 'r') as fh:
        for idx,line in enumerate(tqdm(fh, ncols = 100)):
            if '\t' not in line and '|' in line:
                if '|a|' in line:
                    parts = line.split('|')
                    paragraphs.append(parts[-1])
                    curr_pmid.append(parts[0].strip())
                else :
                    parts = line.split('|')
                    titles.append(parts[-1])


    print(len(paragraphs))

    with open(processed_file_path, 'a') as fh:
        for para in zip():
                # fh.write("{}\t{}\n".format(token, tag))
                fh.write(f'{para}\n')


    text = []
    for i in range(0, len(paragraphs)):
        text.append(titles[i] + " " + paragraphs[i])
    
    return curr_pmid, text

def embeddings():
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')
    embeddings = model.encode(sentences)
    print(embeddings)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pmid_file', type=str, required=False)#____used for MedMentions_____
    # parser.add_argument('--processed_file_path', type=str, required=False)

    args = parser.parse_args()

    #tokenizer for tokenizing the dataset
    nlp = spacy.load('en_core_web_trf')
    tokenizer = nlp.tokenizer

    #for preprocessing the text
    
    dataset_name = (args.dataset).split('/')[1]
    print("hereerer")
    extract_paragraph(args.dataset)
    embeddings()
if __name__ == '__main__':
    main()

