import os
import re

from common.utils import read_tsv

def build_data(fns, mode):
    in_fn = fns["input"][mode]
    to_fn = fns["output"][mode]

    data = read_tsv(in_fn)

    case_ids = []
    bio_data = []
    for ex_idx, ex in enumerate(data):
        case_id = ex[0]
        tagged_sentence = ex[1]

        label = ["O" for idx in range(len(tagged_sentence))]
        char_sentence = [ch for ch in tagged_sentence]

        matchObj_iter = re.finditer('(<([^>]+)>([^>]+)<([^>]+)>)', tagged_sentence)
        # matchObj_iter = re.finditer('<[A-Za-z\/][^>]*>', tagged_sentence)
        for matchObj in matchObj_iter:
            tag_span = matchObj.span()
            tag_start_idx = tag_span[0]
            tag_end_idx = tag_span[1] # 실제 end의 한 칸 뒤임

            try:
                tag_value_start_idx = re.search('(<([^>]+)>)', matchObj.group()).end()
                tag_value_end_idx = re.search('(</([^>]+)>)', matchObj.group()).start() - 1
            except:
                print(matchObj.group()[1:tag_value_start_idx-1])
                print(char_sentence)
                print(label)
                print("mode: {}".format(mode))
                raise ValueError(ex_idx)

            tag_name = matchObj.group()[1:tag_value_start_idx-1]

            tag_value_start_idx = tag_value_start_idx + tag_start_idx
            tag_value_end_idx = tag_value_end_idx + tag_start_idx

            for idx in range(tag_start_idx, tag_end_idx):
                label[idx] = "X"
            if tag_name == "weapon" or tag_name == "injured.part":
                label[tag_value_start_idx] = "O"
            else:
                label[tag_value_start_idx] = "B-" + tag_name
                for idx in range(tag_value_start_idx+1, tag_value_end_idx+1):
                    label[idx] = "I-" + tag_name


            end_tag = re.search('(</([^>]+)>)', matchObj.group()).group()
            end_tag_name = end_tag[2:len(end_tag)-1]

            assert (tag_name == end_tag_name), "tag name is not equal at row {} ({}, {})".format(ex_idx+1, tag_name, end_tag_name)

        final_label = []
        final_sentence = []
        for idx in range(len(label)):
            if label[idx] == "X":
                continue
            final_label.append(label[idx])
            final_sentence.append(char_sentence[idx])

        case_ids.append(case_id)
        bio_data.append((final_sentence, final_label))

    with open(to_fn, "w", encoding='utf-8') as f:
        for ex_idx, (text, label) in enumerate(bio_data):
            print("{}".format(case_ids[ex_idx]), file=f)
            for idx in range(len(label)):
                print("{}\t{}".format(text[idx], label[idx]), file=f)
            print("----", file=f)
    print("[{}] BIO Data is dumped at {}".format(mode, to_fn))


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