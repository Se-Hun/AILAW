# ----------------------- Directory ---------------------------- #
import os


def prepare_dir(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def exist_dir(dir_name):
    if not os.path.exists(dir_name):
        return False
    else:
        return True

