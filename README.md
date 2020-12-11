# AILAW
졸업 프로젝트 : 판례 내 주요정보 추출 및 추론 기술에 대한 연구

## 필수 라이브러리 설치
* `conda install scikit-learn`
* `conda install openpyxl`
* `conda install pandas`
* `conda install matplotlib`
* `conda install selenium`
* `conda install xlsxwriter`
* `conda install xlrd`
* `conda install nltk`
* `pip install seqeval`
* pytorch 설치 : [Pytorch 설치 링크](https://pytorch.org/get-started/locally/)
* transformers 설치 : `pip install transformers`
* pytorch-lightning 설치 : `pip install transformers`

## 데이터 전처리
### 정보 추출을 위한 데이터 셋 : CSIE(Case Information Extraction)
data/ner 경로로 이동후, `python 99_build_all_ner.py` 명령어를 실행한다.

### 정보 추론을 위한 데이터 셋 : CSII(Case Information Inference)
data/classification 경로로 이동후, `python 99_build_all_classification.py` 명령어를 실행한다.

## 연속적 레이블링을 통한 범죄 사실 내 정보 추출 실험
`python run_ner_pl.py --do_train --do_predict --text_reader bert --max_seq_length 512 --batch_size 8 --gpu_id 0`와 같은 명령어를 통해 실험을 진행한다.

## 문장 분류를 통한 범죄 사실 내 정보 추론 실험
`python run_classification_pl.py --do_train --do_predict --text_reader bert --max_seq_length 512 --batch_size 8 --gpu_id`와 같은 명령어를 통해 실험을 진행한다.

