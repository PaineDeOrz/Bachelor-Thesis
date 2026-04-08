# Instructions
This directory contains the code and supporting material for the bachelor’s thesis \*\*“Towards Human-Aligned Chess Engines: Replicating Human Player Behavior across Different Skill Levels.”\*\*

## Directory structure
```text
├── scripts/

├── Maia\_dataset/

├── Accuracy\_test/

├── Maia2/

├── GAN/

```

## What each folder contains

### `scripts/`
Contains cluster submission scripts used during development and is specific to the original university environment (local paths, cluster-specific commands etc.).

These files are included as documentation of the research workflow and may need to be adapted before reuse on another system.

### `Maia\_dataset/`
This folder is expected to contain the Maia testing dataset used for evaluation.

Before running the evaluation scripts, download the Maia test set from:

[Datasets | Computational Social Science Lab](http://csslab.cs.toronto.edu/datasets/)

### `Accuracy\_test/`
Contains the evaluation script and example output for the accuracy comparison experiment.

Files in this folder:

- `accuracy\_test.py` — script used to compare engine predictions across rating bins.

- `Stockfish vs Leela vs Maia 100,500.png` — example result image produced by running the script with those parameters.

### `Maia2/`
**Behavior Stylometry** analysis - K-Means Clustering + PCA in `cluster.py`, accuracy measurement and plotting in `accuracy_plot.py`

### `GAN/`
**Discriminator**'s entire pre-training pipeline and model in `discriminator_dataset.py`, `discriminator_model.py` and 'discriminator_train.py`

**WGAN training pipeline** explained in `main.py` and `train.py`

### `requirements.txt`
All Python dependencies for reproducibility.

## Quick Start
1. `pip install -r requirements.txt`
2. Download Lichess PGN data: [database.lichess.org](https://database.lichess.org/)
3. Place Maia test set in `Maia_dataset/`
4. Run analysis scripts from each folder

## Reproducibility
The repository is meant to show both the thesis results and the workflow used to produce them.  

The included scripts and example outputs should make it easier to understand how the experiments were run and how the final figures were generated.