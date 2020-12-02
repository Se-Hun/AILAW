"""
    Build All Data for Experiments

    Author : Sangkeun Jung (2020)
"""

import os

if __name__ == '__main__':
    run_dir = os.path.join("./", "run")
    if not os.path.exists(run_dir): os.makedirs(run_dir)

    cmd = 'python 1_split_data.py'
    print("Execute : ", cmd)
    os.system(cmd)

    cmd = 'python 2_map_id_to_label.py'
    print("Execute : ", cmd)
    os.system(cmd)

    cmd = 'python 3_build_vocab.py'
    print("Execute : ", cmd)
    os.system(cmd)
