import os
import json

import pandas as pd

from common.utils import prepare_dir

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
            "paragraphs" : []
        })
        for features in feature_df.iterrows():
            paragraph = {}
            for feature_name in feature_names:
                paragraph[column_names_to_keys[feature_name]] = features[1].loc[feature_name]

            cases[data_idx]["paragraphs"].append(paragraph)

    return cases

def merge_texts_and_tags(cases, ner_tags):
    tagged_cases = []
    for data_idx, case in enumerate(cases):
        case_id = case["id"]
        case_title = case["title"]
        tagged_cases.append({
            "id": case_id,
            "title": case_title,
            "paragraphs" : []
        })

        paragraphs = case["paragraphs"]
        new_paragraphs = []
        valid_bool = False
        for paragraph in paragraphs:
            tagging_list = [
                {
                    "start" : int(paragraph[tag_name + "/start"]),
                    "end" : int(paragraph[tag_name + "/start"]) + len(paragraph[tag_name]),
                    "word" : paragraph[tag_name],
                    "tag" : tag_name
                } for tag_name in ner_tags]
            tagging_list = sorted(tagging_list, key=lambda x: x["start"])

            target_text = paragraph["text"]
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
                if (valid_word != tag_info['word']):
                    valid_bool = True
                # assert (valid_word == tag_info['word']), "Nested Tag Or Error In Case ID {}".format(case_id)

        if valid_bool:
            print("Nested Tag Or Error In Case ID {}".format(case_id))

            new_paragraphs.append(target_text)
        tagged_cases[data_idx]["paragraphs"] = new_paragraphs

    return tagged_cases

def apply_sentence_splitter(cases):
    import kss # korean sentence splitter

    spltted_cases = []
    for data_idx, case in enumerate(cases):
        case_id = case["id"]
        case_title = case["title"]
        spltted_cases.append({
            "id": case_id,
            "title": case_title,
            "paragraphs" : []
        })

        paragraphs = case["paragraphs"]
        new_paragraphs = []
        for paragraph in paragraphs:
            sentences = []
            for sentence in kss.split_sentences(paragraph):
                sentences.append(sentence)
            new_paragraphs.append(sentences)
        spltted_cases[data_idx]["paragraphs"] = new_paragraphs

    return spltted_cases

def build_data(fns, column_names_to_keys, ner_tags):
    in_fn = fns['input']
    to_fn = fns['output']

    df = pd.read_excel(in_fn, sheet_name = '시트1')

    # TODO : (1) 필요한 데이터 excel에서 파싱하기
    cases = parse_dataframe(df, column_names_to_keys)

    # TODO : (2) 각 문장에 태그 부착하기
    tagged_cases = merge_texts_and_tags(cases, ner_tags)

    # TODO : (3) 문장 splitter 적용하기
    spltted_cases = apply_sentence_splitter(tagged_cases)

    # TODO : (4) JSON File에 저장
    with open(to_fn, 'w', encoding='utf-8') as f:
        json.dump(spltted_cases, f, indent=4, ensure_ascii=False)
        print("[Train] Show And Tell Token data is dumped at  ", to_fn)

    # TODO : (optional-4) Text File에 저장 --> 판례별로 구분자 "------" 넣어주기

def apply_sentence_splitter_new(cases, ner_tags):
    import kss  # korean sentence splitter

    splitted_cases = []
    for data_idx, case in enumerate(cases):
        case_id = case["id"]
        case_title = case["title"]
        splitted_cases.append({
            "id": case_id,
            "title": case_title,
            "paragraphs": case["paragraphs"],
            "sentences" : []
        })

        paragraphs = case["paragraphs"]
        new_sentences = []
        global_sentence_idx = 0
        for paragraph in paragraphs:
            paragraph_text = paragraph["text"]

            tagging_list = [
                {
                    "start": int(paragraph[tag_name + "/start"]),
                    "end": int(paragraph[tag_name + "/start"]) + len(paragraph[tag_name]),
                    "word": paragraph[tag_name],
                    "tag": tag_name
                } for tag_name in ner_tags]
            tagging_list = sorted(tagging_list, key=lambda x: x["start"]) # for algorithm speed

            # sentences = []
            merged_sentence_length = 0
            minus_index = 0
            for sentence in kss.split_sentences(paragraph_text):
                merged_sentence_length = merged_sentence_length + len(sentence) + 1

                new_sentences.append({
                    "text" : sentence,
                    "tag_info" : []
                })

                for tag_info in tagging_list:
                    if tag_info["end"] <= merged_sentence_length:
                        tagging_list.remove(tag_info)

                        tag_info["start"] = tag_info["start"] - minus_index
                        tag_info["end"] = tag_info["end"] - minus_index
                        new_sentences[global_sentence_idx]["tag_info"].append(tag_info)
                minus_index = minus_index + len(sentence) + 1
                global_sentence_idx = global_sentence_idx + 1

            # new_paragraphs.append(sentences)
        splitted_cases[data_idx]["sentences"] = new_sentences

    return splitted_cases

def merge_texts_and_tags_new(cases):
    merged_cases = []
    for data_idx, case in enumerate(cases):
        merged_cases.append({
            "id": case["id"],
            "title": case["title"],
            "paragraphs": case["paragraphs"],
            "sentences": case["sentences"],
            "tagged_sentences" : []
        })

        sentences = case["sentences"]
        tagged_sentences = []
        for sentence in sentences:
            if len(sentence["tag_info"]) == 0:
                tagged_sentences.append(sentence["text"])
            else:
                plus_index = 0
                sentence_text = sentence["text"]
                for tag_info in sentence["tag_info"]:
                    start_idx = tag_info['start'] + plus_index
                    start_tag = "<" + tag_info['tag'] + ">"
                    sentence_text = sentence_text[:start_idx] + start_tag + sentence_text[start_idx:]
                    plus_index = plus_index + len(start_tag)

                    end_idx = tag_info['end'] + plus_index
                    end_tag = "</" + tag_info['tag'] + ">"
                    sentence_text = sentence_text[:end_idx] + end_tag + sentence_text[end_idx:]
                    plus_index = plus_index + len(end_tag)

                tagged_sentences.append(sentence_text)
        merged_cases[data_idx]["tagged_sentences"] = tagged_sentences

    return merged_cases

def build_data_new(fns, column_names_to_keys, ner_tags):
    in_fn = fns['input']
    to_fn = fns['output']

    df = pd.read_excel(in_fn, sheet_name = '시트1')

    # TODO : (1) 필요한 데이터 excel에서 파싱하기
    cases = parse_dataframe(df, column_names_to_keys)
    # print(cases)

    # TODO : (2) 문장 Splitter 적용하기
    splitted_cases = apply_sentence_splitter_new(cases, ner_tags)

    # TODO : (3) 각 문장에 태그 부착하기
    merged_cases = merge_texts_and_tags_new(splitted_cases)
    # print(merged_cases)

    # TODO : (4) JSON File에 저장
    with open(to_fn, 'w', encoding='utf-8') as f:
        json.dump(merged_cases, f, indent=4, ensure_ascii=False)
        print("NER data is dumped at  ", to_fn)

if __name__ == '__main__':
    to_folder = os.path.join("./", "run")
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join("./", "original", "law.xlsx"),
        # "output": os.path.join(to_folder, "ner_tag.txt")
        "output" : os.path.join(to_folder, "law_ner_tag.json")
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

    # build_data(fns, column_names_to_keys, ner_tags)
    build_data_new(fns, column_names_to_keys, ner_tags)
