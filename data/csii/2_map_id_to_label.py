import os

from common.utils import prepare_dir

def map_id_to_label(fns, label_map, mode):
    import pandas as pd
    df = pd.read_csv(fns["input"][mode], sep='\t')

    case_ids = []
    ex_texts = []
    ex_labels = []
    for row_idx, (index, row) in enumerate(df.iterrows()):
        case_ids.append(row[0])
        ex_texts.append(row[1])
        ex_labels.append(label_map[row[2]])

    print("Number of Examples : {}".format(len(case_ids)))
    items = {"case_id": case_ids, "text": ex_texts, "label": ex_labels}

    new_df = pd.DataFrame(items)

    # dump as tsv file
    to_fn = fns["output"][mode]
    new_df.to_csv(to_fn, index=False, sep="\t")
    print("data is dumped at ", to_fn)


if __name__ == '__main__':
    data_folder = os.path.join("./", "run")
    prepare_dir(data_folder)

    fns = {
        "input" : {
            "train" : os.path.join(data_folder, "train.tsv"),
            "dev" : os.path.join(data_folder, "dev.tsv")
        },
        "output": {
            "train" : os.path.join(data_folder, "train.tsv"),
            "dev" : os.path.join(data_folder, "dev.tsv")
        }
    }

    label_list = ["이웃주민", "동료/동업", "부부", "친족관계", "연인", "스승-제자",
                  "손님-점원", "형사-피의자", "기타:군대", "기타:종교 관련", "기타:지인", "낯선 사람", "알 수 없음"]

    label_map = {i+1: label for i, label in enumerate(label_list)}

    map_id_to_label(fns, label_map, "train")
    map_id_to_label(fns, label_map, "dev")