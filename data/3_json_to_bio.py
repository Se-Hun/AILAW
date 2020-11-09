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

import os
import json

from common.utils import prepare_dir

def build_data(fns, mode):
    in_fn = fns["input"][mode]
    to_fn = fns["output"][mode]

    bio_data = []
    with open(in_fn, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for idx, ex in enumerate(data):
            sentences = ex["tagged_sentences"]

            for sentence in sentences:
                bio_data.append(sentence)

    with open(to_fn, "w", encoding='utf-8') as f:
        for text in bio_data:
            # TODO : bio로 바꿔서 넣을 것 <-- 현재는 태그 형태로 해둠!
            print(text, file=f)
    print("[{}] BIO Data is dumped at {}".format(mode, to_fn))


if __name__ == '__main__':
    to_folder = os.path.join("./", "run")
    prepare_dir(to_folder)

    fns = {
        "input": {
            "train" : os.path.join("./", "run", "train_ner_data.json"),
            "test" : os.path.join("./", "run", "test_ner_data.json")
        },
        "output": {
            "train" : os.path.join(to_folder, "train_bio_data.txt"),
            "test" : os.path.join(to_folder, "test_bio_data.txt")
        }
    }

    build_data(fns, 'train')
    build_data(fns, 'test')