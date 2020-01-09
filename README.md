# ml-toolkit

## Motivation
The goal is for this repository to contain a series of clean ML examples of different levels of understanding that I can draw from and use as examples, test models, etc. I essentially want to gather all of the best-practice items I find or have used for later reuse.

As a rule, everything must be modular and easily imported and exported for later use.

Thus, everything should be grouped by workflow, but also by use cases!

The rough evaluation metric for this repo's success is how fast I can start working on a Kaggle challenge after downloading the data: getting insights on the data, its distributions, running baseline and finetuning models, getting loss curves and plots.1

The goal is not for this to be some ML framework built on PyTorch, but to focus on a single researcher/developer workflow and make it very easy to begin working. Great for Kaggle competitions, simple data exploration, or experimenting with different models.

The problem is that Pytorch examples are not nearly similar enough. As with most data exploration, you tend to want the ability to modify everything while making everything as transparent as possible. Thus, this is for the developer, the aspiring ML scientist, who wants to learn how to modify a generic template code into a fully-functioning ML pipeline, with consistent styling, an opinionated way of handling logging and metrics, and a simple way of checkpointing and resuming progress.

Distributed pytorch builtin? Maybe sometime in the far future.


# Current Workflow
1. Run `train.py`, which saves model checkpoints, output predictions, and tensorboards in the same folder.
2. Start tensorboard using the `checkpoints/` folder
3. Run `test.py` to get final predictions.

This repo draws from learning from:
- PyTorch examples (best practices)
- landmark
- self-driving
- aptos-blindness
- cs230, 231n, 224n, 398


# Directory Structure
models/ contains 


# TODO
1. Write an init script that copies all of example/ into the repo
2. Allow customizability with config.json


# Directory Structure
- checkpoints/ (Only created once you run train.py)
- data/
- models/
    - layers/
- .gitignore
- dataset.py (Stub)
- models.py
- preprocess.py (R) (Stub)
- test.py (R)
- train.py (R)
- viz.py (R)


# Goal Workflow
1. Run `torch init` to initialize a repo with a given init.config.
2. Copy data into `data/`.
3. Fill in `preprocess.py` and `dataset.py`. (Optional: explore data with `visualize.py`)
4. Change `const.py` to specify input/output dimensions, batch size, etc.
5. Run `train.py`, which saves model checkpoints, output predictions, and tensorboards in the same folder. Also automatically starts tensorboard server in a tmux session.
6. Run `test.py` to get final predictions.
