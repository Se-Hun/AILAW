# AILAW
졸업 프로젝트

## 필수 라이브러리 설치
* `conda install scikit-learn`
* `conda install openpyxl`
* `conda install pandas`
* `conda install selenium`
* `conda install xlsxwriter`
* `conda install xlrd`
* `conda install nltk`
* `pip install seqeval==0.0.5`
* pytorch 설치
* transformers 설치

## 가해자-피해자 간의 관계 분류 실험
`python run_classifier.py --do_eval --model_type bert --max_epoch 8 --batch_size 8`

## 범죄사실에서 중요 정보 추출 실험
`python run_ner.py --data_dir=data/ner/run/ --bert_model=bert-base-multilingual-cased --task_name=ner --output_dir=model/ner/ --max_seq_length=128 --num_train_epochs 3 --do_train --do_eval --warmup_proportion=0.1`
