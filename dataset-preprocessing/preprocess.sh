# if [ ! -d resources/BC5CDR ];
# then
#     mkdir resources/BC5CDR
# fi 

# # echo "Creating dicts for BC5CDR"

# # python3 get_dict.py \
# #     --dataset datasets/BC5CDR/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt \
# #     --mention_dict_path resources/BC5CDR/mention_dict.txt \
# #     --cui_dict_path resources/BC5CDR/cui_dict.txt \
# #     --sample_fraction 0.1 

# echo "Converting BC5CDR dataset to BIO tag format"

# if [ -f resources/BC5CDR/train.txt ]; then
#     rm resources/BC5CDR/train.txt
# fi

# if [ -f resources/BC5CDR/devel.txt ]; then
#     rm resources/BC5CDR/devel.txt
# fi

# if [ -f resources/BC5CDR/test.txt ]; then
#     rm resources/BC5CDR/test.txt
# fi


# for fr in 10 20 30 40 50 60 70 80 90
# do
# 	echo "start for : $fr"
		
# 	python3 preprocess.py \
# 			--dataset datasets/BC5CDR/train_corpus.txt \
# 			--processed_file_path resources/BC5CDR/IF/train_$fr.txt \
# 			--sample_fraction $fr
				
# 	echo " - Training set $fr is done"
		
# done


# python3 preprocess.py \
#     --dataset datasets/BC5CDR/train_corpus.txt \
#     --processed_file_path resources/BC5CDR/train_100.txt \
#     --sample_fraction 100 

# echo " - All Training set $fr is done"

# python3 preprocess.py \
#     --dataset datasets/BC5CDR/dev_corpus.txt \
#     --processed_file_path resources/BC5CDR/devel.txt \
#     --sample_fraction 100 

# echo " - Development set done"

# python3 preprocess.py \
#    --dataset datasets/BC5CDR/test_corpus.txt \
#    --processed_file_path resources/BC5CDR/test.txt \
#    --sample_fraction 100 

# echo " - Test set done"



if [ ! -d resources/ ];
then
    mkdir resources 
fi 

if [ ! -d resources/MedMentions ];
then 
    mkdir resources/MedMentions 
fi 

# echo "Creating dicts for MedMentions."
# python3 get_dict.py \
#     --dataset datasets/MedMentions/full/data/corpus_pubtator.txt \
#     --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_trng.txt \
#     --mention_dict_path resources/MedMentions/mention_dict.txt \
#     --cui_dict_path resources/MedMentions/cui_dict.txt 


# echo "Converting Medmention dataset to BIO tag format:"

# if [ -f resources/MedMentions/train.txt ]; then
#     rm resources/MedMentions/train.txt
# fi 

# if [ -f resources/MedMentions/devel.txt ]; then
#     rm resources/MedMentions/devel.txt
# fi

# if [ -f resources/MedMentions/test.txt ]; then
#     rm resources/MedMentions/test.txt
# fi


# 1-R, 2-B, 3-C
for t in 2 3
do
    for fr in 10 20 30 40 50 60 70 80 90
    do	
        echo "start for: $fr"

        python3 preprocess.py \
            --dataset datasets/MedMentions/full/data/corpus_pubtator.txt \
            --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_trng.txt \
            --processed_file_path resources/MedMentions/IF/$t/train_$fr.txt \
            --sample_fraction $fr \
            --sample_type $t
            
        echo " - Training set $fr done"

    done
done
echo " - Training set done"


# python3 preprocess.py \
# 	--dataset datasets/MedMentions/full/data/corpus_pubtator.txt \
# 	--pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_trng.txt \
# 	--processed_file_path resources/MedMentions/train_100.txt \
# 	--sample_fraction 100

# python3 preprocess.py \
#      --dataset datasets/MedMentions/full/data/corpus_pubtator.txt \
#      --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_dev.txt \
#      --processed_file_path resources/MedMentions/devel.txt \
#      --sample_fraction 100

#  echo " - Development set done"

#  python3 preprocess.py \
#      --dataset datasets/MedMentions/full/data/corpus_pubtator.txt \
#      --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_test.txt \
#      --processed_file_path resources/MedMentions/test.txt \
#      --sample_fraction 100

#  echo " - Test set done"

