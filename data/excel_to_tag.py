import os

import pandas as pd
import numpy as np

# file_path = "../korcl_2019/law.xlsx"
# law_df = pd.read_excel(file_path, sheet_name = 'Sheet1')
#
# answer_columns = law_df.columns[3:]
# for idx, col in enumerate(answer_columns) :
#     if idx % 2 == 0 :
#         law_df[col] = law_df[col].apply(lambda x: len(x))
#
# law_df['인덱스들'] = "["+law_df[answer_columns].apply(lambda x : ",".join(x.values.astype(str)), axis=1)+"]"
#
#
# tags = ["WHO","WHO","WHEN","WHEN","WHERE","WHERE","WAHT","WAHT","AGE","AGE","RELATION","RELATION"]
# def tagging(text, answers) :
#     answers = answers.replace("UNKNOWN", "-1004")
#     answers = eval(answers)
#
#     ner_tags = [[char, "O"] for char in text]
#     print(ner_tags)
#     for idx in range(1,len(answers), 2) :
#         if answers[idx] > 0 :
#             start = answers[idx]
#             length = answers[idx-1]
#             ner_tags[start][1] = "B" + "-" + tags[idx]
#             for i in range(start+1, start+length) :
#                 ner_tags[i][1] = "I" + "-" + tags[idx]
#
#     return ner_tags
#
# law_df["NER"] = law_df.apply(lambda x : tagging(x["범죄사실"], x["인덱스들"]), axis=1)
#
# with open("ner_tag.txt", "w", -1, "utf-8") as fp :
#     for text in law_df["NER"].tolist() :
#         for char, tag in text :
#             fp.write(char+"\t"+tag+"\n")
#
#         fp.write("------------------------------\n")

def build_tag(fns, ner_tags):
    in_fn = fns['input']
    to_fn = fns['output']

    df = pd.read_excel(in_fn, sheet_name = '시트1')

    #TODO : (1) 필요한 데이터 excel에서 파싱하기
    neccessary_columns = df.columns[3:]
    for idx, col in enumerate(neccessary_columns):
        if idx % 2 == 0:
            df[col] = df[col].apply(lambda x: len(x))

    df['인덱스들'] = "[" + df[neccessary_columns].apply(lambda x: ",".join(x.values.astype(str)), axis=1) + "]"

    # TODO : (2) 문장 splitter 적용하기

    # TODO : (3) 각 문장에 태그 부착하기

    # TODO : (4) Text File에 저장 --> 판례별로 구분자 "------" 넣어주기

if __name__ == '__main__':
    fns = {
        "input": os.path.join("./", "original", "law.xlsx"),
        "output": os.path.join("./", "run", "ner_tag.txt")
    }

    ner_tags = ["victim.who", "crime.when", "crime.where", "crime.what", "victim.age", "victim.attacker.relation"]

    build_tag(fns, ner_tags)