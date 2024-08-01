import os
import shutil


def get_index(filename):
    basename = os.path.basename(filename)
    return int(os.path.splitext(basename)[0])


def is_index_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0].isdigit()


def clear_folder(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)


def get_sorted_filenames_by_index(dirname, isabs=True):
    filenames = sorted(
        list(filter(is_index_filename, os.listdir(dirname))), key=lambda x: get_index(x))
    if isabs:
        filenames = [os.path.join(dirname, filename) for filename in filenames]
    return filenames
