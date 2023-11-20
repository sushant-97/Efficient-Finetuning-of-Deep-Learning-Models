#python3 preprocess.py --dataset datasets/MedMentions/full/data/corpus_pubtator.txt --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_trng.txt --processed_file_path train.txt
#python3 preprocess.py --dataset datasets/MedMentions/full/data/corpus_pubtator.txt --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_dev.txt --processed_file_path devel.txt
#python3 preprocess.py --dataset datasets/MedMentions/full/data/corpus_pubtator.txt --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_test.txt --processed_file_path test.txt
python3 preprocess.py --dataset datasets/BC5CDR/train_corpus.txt --processed_file_path processed/train.txt
#python3 preprocess.py --dataset datasets/BC5CDR/dev_corpus.txt --processed_file_path processed/devel.txt
#python3 preprocess.py --dataset datasets/BC5CDR/test_corpus.txt --processed_file_path processed/test.txt
