# # #!/bin/bash
# # echo $(pwd)
# # echo $(pwd)
# # echo $(pwd)
# # export DATA=MedMentions
# # export DATA_DIR="MedMentions_MODEL_vanilla"
# # mkdir -p "$DATA_DIR"
# # export PRED_DIR="MedMentions_PRED_vanilla"
# # mkdir -p "$PRED_DIR"
# # export SEED=1


# #!/bin/bash
# echo $(pwd)
# echo $(pwd)
# echo $(pwd)
# export DATA=BC5CDR
# export DATA_DIR="BC5CDR_MODEL_vanilla"
# mkdir -p "$DATA_DIR"
# export PRED_DIR="BC5CDR_PRED_vanilla"
# mkdir -p "$PRED_DIR"
# export SEED=1


# # In train_vanilla.py
# for try in 1 2
# do
#     sh preprocess.sh    # CREATE NEW SUBSETS
    
#     # To do training on BC5CDR Disease dataset
#     python3 train_vanilla.py --dataset_name $DATA --output_model_directory $DATA_DIR --output_tokenizer_directory $DATA_DIR

#     #To do evaluatoin on testing data
#     python3 test.py \
#             --dataset_name $DATA \
#             --model_directory $DATA_DIR \
#             --tokenizer_directory $DATA_DIR \
#             --output_file $PRED_DIR/BP_test_preds_BC5CDR \
#             --model_type vanilla

# done

#!/bin/bash
echo $(pwd)
echo $(pwd)
echo $(pwd)
export DATA=BC5CDR
export DATA_DIR="BC5CDR_MODEL_vanilla"
mkdir -p "$DATA_DIR"
export PRED_DIR="BC5CDR_PRED_vanilla"
mkdir -p "$PRED_DIR"
export SEED=1


# In train_vanilla.py

# To do training on BC5CDR Disease dataset
# python3 train_vanilla.py --dataset_name $DATA --output_model_directory $DATA_DIR --output_tokenizer_directory $DATA_DIR

#To do evaluatoin on testing data
python3 test_.py \
         --dataset_name $DATA \
         --model_directory $DATA_DIR \
         --tokenizer_directory $DATA_DIR \
         --output_file $PRED_DIR/BP_test_preds_BC5CDR \
         --model_type vanilla






