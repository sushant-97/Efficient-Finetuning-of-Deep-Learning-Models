#!/bin/bash
echo $(pwd)
export DATA=BC5CDR
export DATA_DIR=BC5CDR_model/BestModel
mkdir -p "$DATA_DIR"
export PRED_DIR=BC5CDR_PRED
mkdir -p "$PRED_DIR"
export SEED=1

# To download pretrained BERT from transformers
#python trans.py 

# To do training on MEDCDR Disease dataset ++++++  
# python3 train.py --dataset_name $DATA --bias_file $DATA/Bias/new_bias.txt --teacher_file $DATA/BioBERTCRF/main_prob_dist.txt --loss_fn Plain --output_model_directory $DATA_DIR/BC5S --output_tokenizer_directory $DATA_DIR/BC5S

#To do tmp_evaluatoin on testing data
python3 test.py --dataset_name $DATA --test_file_name $DATA/test.txt --model_directory $DATA_DIR --tokenizer_directory $DATA_DIR --output_file $PRED_DIR/vannila_BC5CDR.txt

python3 eval_entity_level.py --dataset_name $DATA --gold_file $DATA/test.txt --pred_file $PRED_DIR/vannila_BC5CDR.txt


# python3 split.py --mention_dictionary $DATA/mention_dict.txt --cui_dictionary $DATA/cui_dict.txt --gold_labels $DATA/test.txt --gold_cuis $DATA/test_cuis.txt --predictions $PRED_DIR/vannila_BC5CDR.txt

# #To split data in Mem, Syn and Con and store the data in separate file
# python3 splitData.py --mention_dictionary $DATA/mention_dict.txt --cui_dictionary $DATA/cui_dict.txt --dataset_name $DATA

# #To generate prediction file for Mem, Syn and Con data
# python3 test.py --dataset_name $DATA --test_file_name $DATA/test_mem.txt --model_directory $DATA_DIR --tokenizer_directory $DATA_DIR --output_file $PRED_DIR/vannila_BC5CDR_mem.txt

# python3 test.py --dataset_name $DATA --test_file_name $DATA/test_syn.txt --model_directory $DATA_DIR --tokenizer_directory $DATA_DIR --output_file $PRED_DIR/vannila_BC5CDR_syn.txt

# python3 test.py --dataset_name $DATA --test_file_name $DATA/test_con.txt --model_directory $DATA_DIR --tokenizer_directory $DATA_DIR --output_file $PRED_DIR/vannila_BC5CDR_con.txt

# #To evaluate the prediction file or Mem, Syn and Con data
# python3 eval_entity_level.py --dataset_name $DATA --gold_file $DATA/test_mem.txt --pred_file $PRED_DIR/vannila_BC5CDR_mem.txt

# python3 eval_entity_level.py --dataset_name $DATA --gold_file $DATA/test_syn.txt --pred_file $PRED_DIR/vannila_BC5CDR_syn.txt

# python3 eval_entity_level.py --dataset_name $DATA --gold_file $DATA/test_con.txt --pred_file $PRED_DIR/vannila_BC5CDR_con.txt
