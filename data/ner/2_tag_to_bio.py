import os
import re

from common.utils import read_tsv

def build_data(fns, mode):
    in_fn = fns["input"][mode]
    to_fn = fns["output"][mode]

    data = read_tsv(in_fn)

    bio_data = []
    for ex in data:
        tagged_sentence = ex[1]

        label = ["O" for idx in range(len(tagged_sentence))]
        char_sentence = [ch for ch in tagged_sentence]

        matchObj_iter = re.finditer('(<([^>]+)>([^>]+)<([^>]+)>)', tagged_sentence)
        # matchObj_iter = re.finditer('<[A-Za-z\/][^>]*>', tagged_sentence)
        for matchObj in matchObj_iter:
            tag_span = matchObj.span()
            tag_start_idx = tag_span[0]
            tag_end_idx = tag_span[1] # 실제 end의 한 칸 뒤임

            tag_value_start_idx = re.search('(<([^>]+)>)', matchObj.group()).end()
            tag_value_end_idx = re.search('(</([^>]+)>)', matchObj.group()).start() - 1

            tag_name = matchObj.group()[1:tag_value_start_idx-1]

            tag_value_start_idx = tag_value_start_idx + tag_start_idx
            tag_value_end_idx = tag_value_end_idx + tag_start_idx

            for idx in range(tag_start_idx, tag_end_idx):
                label[idx] = "X"
            label[tag_value_start_idx] = "B-" + tag_name
            for idx in range(tag_value_start_idx+1, tag_value_end_idx+1):
                label[idx] = "I-" + tag_name

        final_label = []
        final_sentence = []
        for idx in range(len(label)):
            if label[idx] == "X":
                continue
            final_label.append(label[idx])
            final_sentence.append(char_sentence[idx])

        bio_data.append((final_sentence, final_label))

    with open(to_fn, "w", encoding='utf-8') as f:
        for text, label in bio_data:
            for idx in range(len(label)):
                print("{}\t{}".format(text[idx], label[idx]), file=f)
            print("----", file=f)
    print("[{}] BIO Data is dumped at {}".format(mode, to_fn))

            # print(char_sentence)
            # print(label)

            # label
            #
            # tag_start = False
            # tag = ''
            # for idx in range(tag_start_idx, tag_end_idx):
            #     ch = char_sentence[idx]
            #     if tag_start and ch != "/" and ch != ">":
            #         label[idx] = "X"
            #         tag = tag + ch
            #         continue
            #     if ch == "<":
            #         label[idx] = "X"
            #         tag_start = True
            #         continue
            #     if ch == ">":
            #         label[idx] = ""
            #     if char_sentence[idx]:
            #         label[idx] = "X"
            #
            #     if char_sentence[idx] == "<":
            #         label[idx] = "X"
            #         tag_start = True
            #         continue
            #     if char_sentence[idx] == ">":
            #         label[idx] = "X"
            #     if tag_start:
            #         tag = tag + char_sentence[idx]
            #         label[idx] = "X"

        # assert (len(label) == len(char_sentence)), "error ? "
        #
        # bio_label = []
        # bio_ch = []
        # is_tag = False
        # tag = ''
        # for idx in range(label):
        #     if label[idx] == "O":
        #         bio_label.append(label[idx])
        #         bio_ch.append(char_sentence[idx])
        #
        #     if label[idx] == "X":
        #         if char_sentence[idx] == "<":
        #             is_tag = True
        #
        #
        #         if is_tag:
        #             continue
        #         else:


        # print(char_sentence)
        # print(label)
            #
            # print(span)
            # print(tagged_sentence[span[0]])
            # print(tagged_sentence[span[1]-1])

            # print(matchObj.group())
            # if matchObj.group()[1] == "/":
            #     end_tag_spans.append(matchObj.span())
            # else:
            #     start_tag_spans.append(matchObj.span())

        # sentence = re.sub('(<([^>]+)>)', '', tagged_sentence)
        # tag_continue = False
        # for ch in tagged_sentence:


        # text = []
        # label = []
        # # prev = ''
        #
        # # tag_start = True
        # # tag_end = False
        # tag_text_idx = []
        # tag_list = []
        #
        # tag = ''
        # tag_start = False
        # tag_start_text_idx = -1
        # for idx, ch in enumerate(tagged_sentence):
        #     if ch == "<":
        #         tag_start = True
        #         continue
        #     if tag_start:
        #         tag = tag + ch
        #         continue
        #     if ch == ">":
        #         tag_start_text_idx = idx + 1
        #         tag_list.append(tag)
        #         tag = ''
        #         continue
        #
        #     if ch == "<":
        #         tag_start = True
        #         prev = ch
        #         continue
        #     if prev == "<" and ch == "/":
        #         tag_end = True
        #         prev = ch
        #         continue
        #     if tag_start and ch != ">":
        #         tag = tag + ch
        #         prev = ch
        #         continue
        #     if tag_start and ch == ">":
        #         prev = ch
        #         continue

        # text = re.sub('<[A-Za-z\/][^>]*>', '', tagged_sentence)

        # test = re.search('(<([^>]+)>)', tagged_sentence)
        # body = re.search('<{.+?}>', tagged_sentence, re.I | re.S)
        # print(body)

        # text = []
        # label = []
        # tag = ''
        # tag_start = False
        # tag_text_start = False
        # tag_end = False
        # for ch in tagged_sentence:
        #     if ch == "<":
        #         tag_start = True
        #         continue
        #     if tag_start and ch != "/":
        #         tag = tag + ch
        #         continue
        #     if ch == ">":
        #         tag_start = False
        #         tag_text_start = True
        #         continue
        #
        #     if tag_text_start:
        #         text.append(ch)
        #         label.append()
        #
        #     text.append(ch)
        #     label.append("O")


    # print(data)


if __name__ == '__main__':
    in_folder = os.path.join("./", "run")
    to_folder = os.path.join("./", "run")

    fns = {
        "input": {
            "train": os.path.join(in_folder, "train.tsv"),
            "dev": os.path.join(in_folder, "dev.tsv")
        },
        "output": {
            "train": os.path.join(to_folder, "train.bio.txt"),
            "dev": os.path.join(to_folder, "dev.bio.txt")
        }
    }

    build_data(fns, "train")
    build_data(fns, "dev")