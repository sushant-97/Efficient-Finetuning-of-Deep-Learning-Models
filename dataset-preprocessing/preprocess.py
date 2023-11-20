import csv
import spacy 
import os
import argparse
import textPreprocessing
from tqdm import tqdm
import random



#___________Approach 3 - Comparing text rather than position________

special_sym = ['-', '+', '(', ')', '[', ']']
# special_sym = ['-', '+', '(', ')']

def convert_dataset_to_BIO_format(nlp, tokenizer, preprocessor, dataset, processed_file_path, t, is_medmention=False, pmid_file_path="", fraction = "100"):
    # delete output file if it exists (needed as append used while writing to file)
    if os.path.exists(processed_file_path):
        os.remove(processed_file_path)
        print(f'{processed_file_path} was removed.')
    given_mentions = []
    sentence = []
    tokenized_sent = []
    tokenized_token = []
    entity = []
    entity_labels = []
    entity_cui_id = []
    bio_tags = []
    curr_pmid = ""
    pmid_lst = []
    mention_id = {}
    if is_medmention:
        with open(pmid_file_path, 'r') as fh:
            for line in fh:
                pmid_lst.append(line.strip())

    subset = sample_data(dataset, fraction, t)
    subset = [str(s) for s in subset]
    
    with open(dataset, 'r') as fh:
        for idx,line in enumerate(tqdm(fh, ncols = 100)):    
            # print(line)
            #line containing title(t) and abstract(a)
            if '\t' not in line and '|' in line:
                parts = line.split('|')
                curr_pmid = parts[0].strip()
                # print(curr_pmid)
                if curr_pmid not in subset:
                    continue
                if len(parts[-1]) > 0:
                    sentence.append(parts[-1].rstrip('\n'))
                # print(curr_pmid)

            # lines containing mentions
            # given_mentions has all mentions for a t+a combine
            if '\t' in line:
                parts = line.split('\t')
                
                if parts[0] not in subset:  ## check if curr_id is in subset
                    continue
                if len(parts) <= 4:
                    continue
                # tmp_str = ""
                # for i in range(3,len(parts)-2):
                #     if tmp_str == "":
                #         tmp_str += parts[i]
                #     else:
                #         tmp_str = tmp_str + ' ' + parts[i]
                # given_mentions.append(tmp_str)
                
                given_mentions.append(parts[3].strip())
                mention_id[parts[3].strip()] = [(parts[4].split(',')[0]).strip(),(parts[5].split(',')[0]).strip()]
                
            # when one t+a finished reading
            if len(line.strip()) == 0:
                # print("********hereherer")
                if is_medmention and curr_pmid not in pmid_lst:
                    given_mentions = []
                    sentence = []
                    tokenized_sent = []
                    tokenized_token = []
                    bio_tags = []
                if len(sentence) == 0:
                    continue
                for s in sentence:
                    doc = nlp(s)
                    for x in doc.sents:
                        tokenized_sent.append(str(x))
                        # if len(x) != 0:
                        #     tokenized_sent.append("")
                

                #tokenizing sentences
                for sent in tokenized_sent:
                    if len(sent.strip()) == 0:
                        tokenized_token.append(sent)
                        continue 
                    words = sent.split(' ')
                    new_sent = ""
                    for word in words:
                        new_word = word
                        for sym in special_sym:
                            if sym in new_word:
                                pos = new_word.find(sym)
                                new_word = new_word[:pos] + ' ' + new_word[pos] + ' ' + new_word[pos+1:]
                        while len(new_word) > 0 and new_word[0] == ' ':
                            new_word = new_word[1:]
                        while len(new_word) > 0 and new_word[-1] == ' ':
                            new_word = new_word[:-1]
                        if len(new_sent) == 0:
                            new_sent = new_word
                        else:
                            new_sent = new_sent + ' ' + new_word

                    temp = tokenizer(new_sent)

                    for x in temp:
                        tokenized_token.append(str(x).strip())
                    tokenized_token.append("") # to signify end of a sentence

                #Adding BIO tokens
                i = 0
                for mention in given_mentions:
                    splited_mention = mention.split()
                    m_len = 0
                    for s_m in splited_mention:
                        m_len += len(s_m)
                    # print(m_len)
                    while i < len(tokenized_token):
                        token_to_match = ""
                        itr = -1
                        our_len = 0
                        steps_count = 0
                        while our_len < m_len:
                            if len(token_to_match) == 0 and tokenized_token[i].strip() == '.':
                                break
                            if len(tokenized_token[i]) > 0 and len(mention) > 0 and preprocessor.run(tokenized_token[i][0].strip()) != preprocessor.run(mention[0].strip()):
                                break
                            if i + itr >= len(tokenized_token):
                                break
                            steps_count += 1
                            itr += 1
                            if token_to_match == "":
                                token_to_match = tokenized_token[i].strip()
                                our_len += len(tokenized_token[i].strip())
                            elif i+itr < len(tokenized_token):
                                token_to_match = token_to_match + ' ' + tokenized_token[i+itr].strip()
                                our_len += len(tokenized_token[i+itr].strip())

                        if preprocessor.run(token_to_match) == preprocessor.run(mention.strip()):
                            class_tag = 'B' #+ mention_id[mention][0]
                            entity.append(mention.strip())
                            entity_labels.append(mention_id[mention][0])
                            entity_cui_id.append(mention_id[mention][1])
                            bio_tags.append(class_tag)
                            for itr in range(steps_count-1):
                                bio_tags.append('I')
                            i += steps_count
                            break
                        elif len(str(tokenized_token[i]).strip()) == 0:
                            bio_tags.append('X')
                            i += 1 
                        else:
                            bio_tags.append('O')
                            i += 1

                while i < len(tokenized_token):
                    bio_tags.append('O')
                    i += 1

                with open(processed_file_path, 'a') as fh:
                    for token, tag in zip(tokenized_token, bio_tags):
                        if token.strip('\n') == "" or tag == 'X':
                            fh.write("\n")
                        else:
                            # fh.write("{}\t{}\n".format(token, tag))
                            fh.write(f'{token}\t{tag}\n')

                given_mentions = []
                sentence = []
                tokenized_sent = []
                tokenized_token = []
                bio_tags = []
                mention_id = {}

    entity_cui = f'entity_cui_{fraction}.txt'
    with open(entity_cui, 'w') as fh:
        for i in range(len(entity)):
            fh.write(f'{entity[i]}\t{entity_labels[i]}\t{entity_cui_id[i]}\n')
    print(len(entity))


def convert_instanceIDS_to_corpusIDS(input_sample_file_path):
	subset = []
	corpusIDS_file = "./input_datafiles/"+"MedMentions"+"/trainCorpus_ids.csv"
	corpusIDS = []
	with open(corpusIDS_file, 'r') as file:
		csvreader = csv.reader(file, delimiter=' ')
		for row in csvreader:
			corpusIDS.append(int(row[0]))
	file.close()
	instanceIDS_file = input_sample_file_path
	instanceIDS = []
	with open(instanceIDS_file, 'r') as file:
		csvreader = csv.reader(file, delimiter=' ')
		for row in csvreader:
			instanceIDS.append(int(row[0]))
	file.close()

	for ids in instanceIDS:
		subset.append(corpusIDS[ids])

  
	return subset, corpusIDS, instanceIDS

def sample_data(dataset, fraction, t):
    dict_ = {"1":"r", "2":"b", "3":"c"}
    input_sample_file_path = "input_datafiles/MedMentions/if_based_samples/" + dict_[t] + fraction + ".csv"
    print(input_sample_file_path)
    s, c, i = convert_instanceIDS_to_corpusIDS(input_sample_file_path)
    print(s)
    return s

# def sample_data(dataset, fraction):
#     ## SAMPLE
#     # creating set of unique Abstract ids
#     unique_ids = set()
#     count = 0
#     with open(dataset, 'r') as file:
#         for line in file:
#             if '\t' not in line and '|' in line:   
#                 count += 1
#                 parts = line.strip().split('|')
#                 record_id = parts[0].strip()
#                 unique_ids.add(record_id)
    
#     unique_ids = list(unique_ids)
#     total = len(unique_ids)
#     to_sample = int(total * fraction)
#     print(len(unique_ids))
#     with open(f'store.txt', 'a') as fh:
#     	fh.write(f'{fraction}\n')
#     	fh.write(','.join(str(i) for i in unique_ids))
#     	fh.write("\n")

#     subset = []
#     while len(subset) < to_sample:
#         # Generate one random number from 0 - len(ids)
#     	sample_ = random.randint(0, total-1)
#     	if sample_ not in subset:
#             subset.append(sample_)
    
#     subset = [unique_ids[x] for x in subset]
#     # with open(f'store.txt', 'a') as fh:
#     # 	fh.write(f'{fraction}\n')
#     # 	fh.write(','.join(str(i) for i in subset))
#     # 	fh.write("\n")

#     return subset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pmid_file', type=str, required=False)#____used for MedMentions_____
    parser.add_argument('--processed_file_path', type=str, required=True)
    parser.add_argument('--sample_fraction', type=str, required=True)
    parser.add_argument('--sample_type', type=str, required=True)

    args = parser.parse_args()

    #tokenizer for tokenizing the dataset
    nlp = spacy.load('en_core_web_trf')
    tokenizer = nlp.tokenizer

    #for preprocessing the text
    preprocessor = textPreprocessing.TextPreprocess()
    dataset_name = (args.dataset).split('/')[1]
    
    
    if dataset_name == 'MedMentions':
        print('MedMentions')
        convert_dataset_to_BIO_format(nlp, tokenizer, preprocessor, args.dataset, args.processed_file_path, t = args.sample_type, is_medmention=True, pmid_file_path=args.pmid_file, fraction = args.sample_fraction)
    else:
        convert_dataset_to_BIO_format(nlp, tokenizer, preprocessor, args.dataset, args.processed_file_path, t = args.sample_type, fraction = args.sample_fraction)

if __name__ == '__main__':
    main()

