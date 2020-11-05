import os
import re

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

def parse_dataframe(df, column_names_to_keys):
    feature_names = list(column_names_to_keys.keys())

    case_ids = df.loc[1:, 'ID'].tolist()
    case_ids = set(case_ids)

    # match test
    for case_id in case_ids:
        target_case_nums = df[(df['ID'] == case_id)]['개수'].tolist()
        for target_num in target_case_nums:
            assert len(df[df['ID'] == case_id]) == target_num, "Case Num is not matched in Case ID {}".format(case_id)

        target_case_titles = df[(df['ID'] == case_id)]['제목'].tolist()
        title_validation = set(target_case_titles)
        assert len(title_validation) == 1, "Case Title is not matched in Case ID {}".format(case_id)

        target_title = target_case_titles[0]
        assert len(df[df['ID'] == case_id]) == len(df[df['제목'] == target_title]), "At same title, Another Case is found at Case ID {}".format(case_id)

    # build Case Data --> JSON
    cases = []
    for data_idx, case_id in enumerate(case_ids):
        case_title = df[df['ID'] == case_id]['제목'].tolist()[0]
        feature_df = df[df['ID'] == case_id][feature_names]

        cases.append({
            "id" : case_id,
            "title" : case_title,
            "sentences" : []
        })
        for features in feature_df.iterrows():
            sentence = {}
            for feature_name in feature_names:
                sentence[column_names_to_keys[feature_name]] = features[1].loc[feature_name]

            cases[data_idx]["sentences"].append(sentence)

    return cases

def merge_texts_and_tags(cases, ner_tags):
    tagged_cases = []
    for data_idx, case in enumerate(cases):
        case_id = case["id"]
        case_title = case["title"]
        tagged_cases.append({
            "id": case_id,
            "title": case_title,
            "sentences" : []
        })

        sentences = case["sentences"]
        new_sentences = []
        for sentence in sentences:
            tagging_list = [
                {
                    "start" : int(sentence[tag_name + "/start"]),
                    "end" : int(sentence[tag_name + "/start"]) + len(sentence[tag_name]),
                    "word" : sentence[tag_name],
                    "tag" : tag_name
                } for tag_name in ner_tags]
            tagging_list = sorted(tagging_list, key=lambda x: x["start"])

            target_text = sentence["text"]
            inserted_string_length = 0
            for tag_info in tagging_list:
                tag_name = tag_info["tag"]

                start_idx = tag_info["start"] + inserted_string_length
                start_tag = "<" + tag_name + ">"
                target_text = target_text[:start_idx] + start_tag + target_text[start_idx:]
                inserted_string_length = inserted_string_length + len(start_tag)

                end_idx = tag_info["end"] + inserted_string_length
                end_tag = "</" + tag_name + ">"
                target_text = target_text[:end_idx] + end_tag + target_text[end_idx:]
                inserted_string_length = inserted_string_length + len(end_tag)

                # valid
                valid_word = target_text[start_idx+len(start_tag):end_idx]
                assert (valid_word == tag_info['word']), "Nested Tag Or Error In Case ID {}".format(case_id)

            new_sentences.append(target_text)
        tagged_cases[data_idx]["sentences"] = new_sentences

    return tagged_cases

def build_data(fns, column_names_to_keys, ner_tags):
    in_fn = fns['input']
    to_fn = fns['output']

    df = pd.read_excel(in_fn, sheet_name = '시트1')

    #TODO : (1) 필요한 데이터 excel에서 파싱하기
    cases = parse_dataframe(df, column_names_to_keys)

    # TODO : (2) 각 문장에 태그 부착하기
    tagged_cases = merge_texts_and_tags(cases, ner_tags)

    print(tagged_cases)

    # TODO : (3) 문장 splitter 적용하기

    # TODO : (4) Text File에 저장 --> 판례별로 구분자 "------" 넣어주기

if __name__ == '__main__':
    fns = {
        "input": os.path.join("./", "original", "law.xlsx"),
        "output": os.path.join("./", "run", "ner_tag.txt")
    }

    column_names_to_keys = {
        "범죄사실" : "text",
        "피해자가 누구인가요?" : "victim.who",
        "피해자_ans" : "victim.who/start",
        "범행이 언제 발생했나요?" : "crime.when",
        "언제_ans" : "crime.when/start",
        "범행이 어디서 발생했나요?" : "crime.where",
        "어디서_ans" : "crime.where/start",
        "어떤 범행이 발생했나요?" : "crime.what",
        "어떤범행_ans" : "crime.what/start",
        # "피해자의 나이가 어떻게 되나요?" : "victim.age",
        # "피해자나이_ans" : "victim.age/start",
        # "피해자와 가해자의 관계가 어떻게 되나요?" : "victim.attacker.relation",
        # "피해자와 가해자 관계_ans" : "victim.attacker.relation/start"
    }
    ner_tags = ["victim.who", "crime.when", "crime.where", "crime.what",
                # "victim.age", "victim.attacker.relation"
                ]

    build_data(fns, column_names_to_keys, ner_tags)