import os
import shutil
import string
import argparse


SOURCE = 'src/'
DESTINATION = 'output_rnn/'
RNN_CONFIG = {
    'presets': {
        'datasets': 'dataset_rnn',
    },
    'substitutions': {
        'loss_fn': 'nn.CrossEntropyLoss()',
        'model': 'BasicRNN'
    }
}

CNN_CONFIG = {
    'presets': {
        'datasets': 'dataset_cnn',
    },
    'substitutions': {
        'loss_fn': 'F.nll_loss',
        'model': 'BasicCNN'
    }
}


def init_project(source, destination, config):
    # Create destination directory if it doesn't exist
    if not os.path.isdir(destination):
        os.makedirs(destination)

    # Copy all files over to the destination directory.
    for filename in os.listdir(source):
        if 'data' in filename:
            continue
        # Fill in template files with entries in config.
        if '_temp.py' in filename:
            result = ''
            with open(os.path.join(source, filename)) as in_file:
                contents = string.Template(in_file.read())
                result = contents.substitute(config['substitutions'])

            new_filename = filename.replace('_temp', '')
            with open(os.path.join(destination, new_filename), 'w') as out_file:
                out_file.write(result)

        else:
            full_src_path = os.path.join(source, filename)
            if os.path.isdir(full_src_path):
                full_dest_path = os.path.join(destination, filename)
                shutil.copytree(full_src_path, full_dest_path)
            else:
                shutil.copy(full_src_path, destination)

    # Copy additional files specified in config, such as dataset.py
    presets = config['presets']
    for key in presets:
        if os.path.isdir(key):
            chosen_file = presets[key]
            if '.py' not in presets[key]:
                chosen_file += '.py'
            key_src_path = os.path.join(key, chosen_file)
            key_dest_path = os.path.join(destination, 'dataset.py')
            shutil.copy(key_src_path, key_dest_path)


def init_pipeline():
    parser = argparse.ArgumentParser(description='PyTorch Project Initializer')

    parser.add_argument('--output_path', type=int, default=100, metavar='N',
                        help='folder to output the project to')

    parser.add_argument('--config_path', type=str, default='',
                        help='filepath to the config.json file')

    parser.add_argument('--visualize', action='store_true', default=True,
                        help='save visualization files')

    return parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    init_project(SOURCE, DESTINATION, RNN_CONFIG)
    main()
