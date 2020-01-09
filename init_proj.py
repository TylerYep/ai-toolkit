import shutil
import os

SOURCE = 'example/'
DESTINATION = 'output/'
CONFIG = {
    'datasets': 'dataset_rnn',
    # 'metrics': 'all',
    # 'viz': 'image',
    # 'models': 'all',
}


def init_project(source, destination, config):
    for file_ in os.listdir(source):
        full_src_path = os.path.join(source, file_)
        if os.path.isdir(full_src_path):
            full_dest_path = os.path.join(destination, file_)
            shutil.copytree(full_src_path, full_dest_path)
        else:
            shutil.copy(full_src_path, destination)

    for key in config:
        if os.path.isdir(key):
            chosen_file = config[key]
            if '.py' not in config[key]:
                chosen_file += '.py'
            key_src_path = os.path.join(key, chosen_file)
            key_dest_path = os.path.join(destination, 'dataset.py')
            shutil.copy(key_src_path, key_dest_path)


if __name__ == '__main__':
    init_project(SOURCE, DESTINATION, CONFIG)
