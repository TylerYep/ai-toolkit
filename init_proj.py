import os
import shutil
import string
import argparse


RNN_CONFIG = {
    'presets': {
        'dataset.py': 'datasets/dataset_rnn.py',
        'viz.py': 'visualizers/viz_rnn.py'
    },
    'substitutions': {
        'loss_fn': 'nn.CrossEntropyLoss()',
        'model': 'BasicRNN'
    }
}

CNN_CONFIG = {
    'presets': {
        'dataset.py': 'datasets/dataset_cnn.py',
        'viz.py': 'visualizers/viz_cnn.py'
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

        full_src_path = os.path.join(source, filename)
        full_dest_path = os.path.join(destination, filename)

        # Remove file if it already exists.
        if filename in os.listdir(destination):
            if os.path.isdir(full_dest_path):
                shutil.rmtree(full_dest_path)
            else:
                os.remove(full_dest_path)

        # Fill in template files with entries in config.
        if '_temp.py' in filename:
            result = ''
            with open(full_src_path) as in_file:
                contents = string.Template(in_file.read())
                result = contents.substitute(config['substitutions'])

            new_dest_path = full_dest_path.replace('_temp', '')
            with open(new_dest_path, 'w') as out_file:
                out_file.write(result)

        # If not a template, copy file over.
        else:
            if os.path.isdir(full_src_path):
                shutil.copytree(full_src_path, full_dest_path)
            else:
                shutil.copy(full_src_path, destination)

    # Copy additional files specified in config, such as dataset.py
    presets = config['presets']
    for key, src_path in presets.items():
        if os.path.isfile(src_path):
            dest_path = os.path.join(destination, key)
            shutil.copy(src_path, dest_path)
        else:
            print("Path not found.")


def init_pipeline():
    parser = argparse.ArgumentParser(description='PyTorch Project Initializer')

    parser.add_argument('project', type=str,
                        help='version of the code to generate')

    parser.add_argument('--output_path', type=int, default=100, metavar='N',
                        help='folder to output the project to')

    parser.add_argument('--config_path', type=str, default='',
                        help='filepath to the config.json file')

    parser.add_argument('--visualize', action='store_true', default=True,
                        help='save visualization files')

    return parser.parse_args()


def main():
    args = init_pipeline()
    if args.project == 'rnn':
        config = RNN_CONFIG
    elif args.project == 'cnn':
        config = CNN_CONFIG
    else:
        raise ValueError


    for filename in os.listdir('source'):
        source = os.path.join('source', filename)
        destination = os.path.join(f'output_{args.project}', filename)
        if os.path.isdir(source):
            init_project(source, destination, config)

        else:
            if filename in os.listdir(f'output_{args.project}'):
                os.remove(destination)
            shutil.copy(source, destination)


if __name__ == '__main__':
    main()
