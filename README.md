# ml-toolkit

## Motivation
When working on ML projects, especially supervised learning, there tends to be a lot of repeated code. This is because in every project, we always want a way to checkpoint our work, visualize loss curves in tensorboard, add additional metrics, and see example output. Some projects we are able to do this better than others. Ideally, we would have some way to consolidate all of this code into a single place.

The problem is that Pytorch examples are not nearly similar enough. Like most data exploration, we want the ability to modify every part of the codebase to handle different loss metrics, different types of data, or different visualizations based on our data dimensions. Combining everything into a single repository would overcomplicate the underlying logic (making the training loop extremely unreadable, for example). We want to strike a balance between extremely minimalistic / readable code that makes it easy to add on extra functionality when needed.

Thus, this project is for developers or ML scientists who want features of a fully-functioning ML pipeline from the beginning. Each project comes with consistent styling, an opinionated way of handling logging, metrics, and checkpointing / resuming training from checkpoints. It also integrates seamlessly with Google Colab and AWS/Google Cloud GPUs.


## Evaluation Criteria
The goal is for this repository to contain a series of clean ML examples of different levels of understanding that I can draw from and use as examples, test models, etc. I essentially want to gather all of the best-practice code gists I find or have used in the past, and make them modular and easily imported or exported for later use.

The goal is not for this to be some ML framework built on PyTorch, but to focus on a single researcher/developer workflow and make it very easy to begin working. Great for Kaggle competitions, simple data exploration, or experimenting with different models.

The rough evaluation metric for this repo's success is how fast I can start working on a Kaggle challenge after downloading the data: getting insights on the data, its distributions, running baseline and finetuning models, getting loss curves and plots.


# Current Workflow
1. Edit `init_proj.py` to your desired configuration.
2. Run `python init_proj.py`, which creates your files in the `output/` directory.
3. Go into the output directory e.g. `cd output/`
3. Depending on your dataset, you may need to paste in your `data/` folder and edit `dataset.py`.
4. Run `train.py`, which saves model checkpoints, output predictions, and tensorboards in the same folder.
5. Start tensorboard using the `checkpoints/` folder with `tensorboard --logdir=checkpoints/`
6. Start and stop training using `python train.py --checkpoint=<checkpoint name>`. The code should automatically resume training at the previous epoch and continue logging to the previous tensorboard.
7. Run `python test.py --checkpoint=<checkpoint name>` to get final predictions.


# Directory Structure
- checkpoints/                  (Only created once you run train.py)
- data/
- metrics/
- models/
    - layers/
    - ...
- visualizers/
- args.py                       (Modify default hyperparameters manually)
- dataset.py                    (Stub)
- metric_tracker.py
- models.py                     (You may opt to keep all your models in one place instead)
- preprocess.py                 (Stub)
- test.py
- train.py
- util.py
- viz.py                        (Stub, create more visualizations if necessary)


# Goal Workflow
1. Run `torch init` to initialize a repo with a given init.config.
2. Copy data into `data/`.
3. Fill in `preprocess.py` and `dataset.py`. (Optional: explore data by running `python visualize.py`)
4. (Maybe?) Change `const.py` to specify input/output dimensions, batch size, etc.
5. Run `train.py`, which saves model checkpoints, output predictions, and tensorboards in the same folder. Also automatically starts tensorboard server in a tmux session.
6. Run `test.py` to get final predictions.


# TODO
- Allow customizability with config.json
