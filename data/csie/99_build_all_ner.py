import os

if __name__ == '__main__':
    run_dir = os.path.join("./", "run")
    if not os.path.exists(run_dir): os.makedirs(run_dir)

    cmd = 'python 1_split_data.py'
    print("Execute : ", cmd)
    os.system(cmd)

    cmd = 'python 2_tag_to_bio.py'
    print("Execute : ", cmd)
    os.system(cmd)

    cmd = 'python 3_build_vocab.py'
    print("Execute : ", cmd)
    os.system(cmd)
