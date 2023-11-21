#!/bin/bash
echo $(pwd)
export DATA=BC5CDR
export DATA_DIR=BC5CDR_model/BestModel
mkdir -p "$DATA_DIR"
export PRED_DIR=BC5CDR_PRED
mkdir -p "$PRED_DIR"
export SEED=1
#fractions = 10 20 30 40 50

for fr in 10 20 30 40 50 60 70 80 90
do  
    echo "for training subset $fr"
    #train
    python3 train.py \
        --dataset_name $DATA \
        --loss_fn Plain \
        --output_model_directory $DATA_DIR/BC5S \
        --output_tokenizer_directory $DATA_DIR/BC5S \
        --sample_fraction $fr
    
    #test
    echo "for testing subset $fr"
    python3 test.py \
    --dataset_name $DATA \
    --test_file_name $DATA/test.txt \
    --model_directory $DATA_DIR/BC5S/$fr \
    --tokenizer_directory $DATA_DIR/BC5S/$fr \
    --output_file $PRED_DIR/vannila_BC5CDR_$fr.txt \
    --sample_fraction $fr

    #eval
    echo "for evaluation subset $fr"
    python3 eval_entity_level.py \
    --dataset_name $DATA \
    --gold_file $DATA/test.txt \
    --pred_file $PRED_DIR/vannila_BC5CDR_$fr.txt \
    --sample_fraction $fr
done


# To download pretrained BERT from transformers
#python savePretrained.py 

# To do training on MEDCDR Disease dataset ++++++  
# python3 train.py --dataset_name $DATA --bias_file $DATA/Bias/new_bias.txt --teacher_file $DATA/BioBERTCRF/main_prob_dist.txt --loss_fn Plain --output_model_directory $DATA_DIR/BC5S --output_tokenizer_directory $DATA_DIR/BC5S
# python3 train.py --dataset_name $DATA --loss_fn Plain --output_model_directory $DATA_DIR/BC5S/train_sets --output_tokenizer_directory $DATA_DIR/BC5S


#To do tmp_evaluatoin on testing data
# python3 test.py --dataset_name $DATA --test_file_name $DATA/test.txt --model_directory $DATA_DIR --tokenizer_directory $DATA_DIR --output_file $PRED_DIR/vannila_BC5CDR.txt

#python3 split.py --mention_dictionary $DATA/mention_dict.txt --cui_dictionary $DATA/cui_dict.txt --gold_labels $DATA/test.txt --gold_cuis $DATA/test_cuis.txt --predictions $PRED_DIR/vannila_BC5CDR.txt

#To split data in Mem, Syn and Con and store the data in separate file
#python3 splitData.py --mention_dictionary $DATA/mention_dict.txt --cui_dictionary $DATA/cui_dict.txt --dataset_name $DATA

#To generate prediction file for Mem, Syn and Con data
#python3 test.py --dataset_name $DATA --test_file_name $DATA/test.txt --model_directory $DATA_DIR --tokenizer_directory $DATA_DIR --output_file $PRED_DIR/vannila_BC5CDR_mem.txt

#python3 test.py --dataset_name $DATA --test_file_name $DATA/test_syn.txt --model_directory $DATA_DIR --tokenizer_directory $DATA_DIR --output_file $PRED_DIR/vannila_BC5CDR_syn.txt

#python3 test.py --dataset_name $DATA --test_file_name $DATA/test_con.txt --model_directory $DATA_DIR --tokenizer_directory $DATA_DIR --output_file $PRED_DIR/vannila_BC5CDR_con.txt

#To evaluate the prediction file or Mem, Syn and Con data
# python3 eval_entity_level.py --dataset_name $DATA --gold_file $DATA/test.txt --pred_file $PRED_DIR/vannila_BC5CDR.txt

#python3 eval_entity_level.py --dataset_name $DATA --gold_file $DATA/test_syn.txt --pred_file $PRED_DIR/vannila_BC5CDR_syn.txt

#python3 eval_entity_level.py --dataset_name $DATA --gold_file $DATA/test_con.txt --pred_file $PRED_DIR/vannila_BC5CDR_con.txt
