# ai-toolkit

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Build Status](https://github.com/TylerYep/ai-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/TylerYep/ai-toolkit/actions/workflows/test.yml)
[![GitHub license](https://img.shields.io/github/license/TylerYep/ai-toolkit)](https://github.com/TylerYep/ai-toolkit/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/TylerYep/ai-toolkit/branch/main/graph/badge.svg)](https://codecov.io/gh/TylerYep/ai-toolkit)

## Motivation

When working on ML projects, especially supervised learning, there tends to be a lot of repeated code, because in every project, we always want a way to checkpoint our work, visualize loss curves in tensorboard, add additional metrics, and see example output. Some projects we are able to do this better than others. Ideally, we want to have some way to consolidate all of this code into a single place.

The problem is that Pytorch examples are generally not very similar. Like most data exploration, we want the ability to modify every part of the codebase to handle different loss metrics, different types of data, or different visualizations based on our data dimensions. Combining everything into a single repository often overcomplicates the underlying logic (making the training loop extremely unreadable, for example). We want to strike a balance between extremely minimalistic / readable code that makes it easy to add extra functionality when needed.

This project is for developers or ML scientists who want features of a fully-functioning ML pipeline from the beginning. Each project comes with consistent styling, an opinionated way of handling logging, metrics, and checkpointing / resuming training from checkpoints. It also integrates seamlessly with Google Colab and AWS/Google Cloud GPUs.

# Try It Out!

The first thing you should do is go into one of the output\_\*/ folders and try training a model.
We currently have the following models:

- MNIST CNN [(Source)](https://github.com/pytorch/examples/blob/main/mnist/main.py)
- Character-Level RNN+LSTM [(Source)](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- Object Detection [(Source)](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

## Notable Features

- In train.py, the code performs some verification checks on all models to make sure you aren't mixing up your batch dimensions.
- Try stopping it and starting it after a couple epochs - it should resume training from the same place.
- On tensorboard, loss curves should already be plotting seamlessly across runs.
- All checkpoints should be available in checkpoints/, which contains activation layers, input data, and best models.
- Scheduling runs is easy by specifying a file in the configs/ folder.

# Evaluation Criteria

The goal is for this repository to contain a series of clean ML examples of different levels of understanding that I can draw from and use as examples, test models, etc. I essentially want to gather all of the best-practice code gists I find or have used in the past, and make them modular and easily imported or exported for later use.

The goal is not for this to be some ML framework built on PyTorch, but to focus on a single researcher/developer workflow and make it very easy to begin working. Great for Kaggle competitions, simple data exploration, or experimenting with different models.

The rough evaluation metric for this repo's success is how fast I can start working on a Kaggle challenge after downloading the data: getting insights on the data, its distributions, running baseline and finetuning models, getting loss curves and plots.

# Current Workflow

1. Add data to your `data/` folder and edit the corresponding DataasetLoader in `datasets/`.
2. Add your config and model to `configs/` and `models/`.
3. Run `train.py`, which saves model checkpoints, output predictions, and tensorboards in the same folder.
4. Start tensorboard using the `checkpoints/` folder with `tensorboard --logdir=checkpoints/`
5. Start and stop training using `python train.py --checkpoint=<checkpoint name>`. The code should automatically resume training at the previous epoch and continue logging to the previous tensorboard.
6. Run `python test.py --checkpoint=<checkpoint name>` to get final predictions.

# Directory Structure

- checkpoints/ (Only created once you run train.py)
- data/
- configs/
- src/
  - datasets/
  - losses/
  - metrics/
  - models/
    - layers/
    - ...
  - visualizations/
  - args.py (Modify default hyperparameters manually)
  - metric_tracker.py
  - test.py
  - train.py
  - util.py
  - verify.py
  - viz.py (Stub, create more visualizations if necessary)
- tests/

# Goal Workflow

1. Move data into `data/`.
2. Fill in `preprocess.py` and `dataset.py`. (Optional: explore data by running `python viz.py`)
3. Change `args.py` to specify input/output dimensions, batch size, etc.
4. Run `train.py`, which saves model checkpoints, output predictions, and tensorboards in the same folder. Also automatically starts tensorboard server in a tmux session. Resume training at any point.
5. Run `test.py` to get final predictions.
