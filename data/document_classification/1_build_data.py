import os

from common.utils import prepare_dir

def build_data(fns):
    in_fn = fns["input"]
    to_fn = fns["output"]

    # TODO : 판례 ID 같은 것끼리 문장 합치기. --> label은 비워두고
    
    # TODO : 이후, 스프레드시트로 옮겨서 다시 태깅하고
    
    # TODO : 2_split_data.py 코드 짜자

if __name__ == '__main__':
    to_folder = "./"
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join("../", "classification", "relation.tsv"),
        "output": os.path.join(to_folder, "doc_relation.tsv")
    }

    build_data(fns)