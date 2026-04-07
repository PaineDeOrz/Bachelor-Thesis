\# Instructions



This directory contains the code and supporting material for the bachelor’s thesis \*\*“Towards Human-Aligned Chess Engines: Replicating Human Player Behavior across Different Skill Levels.”\*\*



\## Directory structure



```text

.

├── scripts/

├── Maia\_dataset/

├── Accuracy\_test/

├── Maia2/

├── GAN/

```



\## What each folder contains



\### `scripts/`

Contains cluster submission scripts used during development and is specific to the original university environment (local paths, cluster-specific commands etc.).



These files are included as documentation of the research workflow and may need to be adapted before reuse on another system.



\### `Maia\_dataset/`

This folder is expected to contain the Maia testing dataset used for evaluation.

Before running the evaluation scripts, download the Maia test set from:



\[Datasets | Computational Social Science Lab](http://csslab.cs.toronto.edu/datasets/)



\### `Accuracy\_test/`

Contains the evaluation script and example output for the accuracy comparison experiment.



Files in this folder:

\- `accuracy\_test.py` — script used to compare engine predictions across rating bins.

\- `Stockfish vs Leela vs Maia 100,500.png` — example result image produced by running the script with those parameters.



\### `Maia2/`





\## Expected setup



To reproduce the experiments, you need:



\- Python 3

\- A working chess-engine environment

\- Stockfish installed locally

\- Leela Chess Zero installed locally

\- Maia weights available locally

\- The Maia testing dataset downloaded into `Maia\_dataset/`





\## How to use the repository



1\. Clone the repository.

2\. Download the Maia test dataset from the link above.

3\. Place the dataset into `Maia\_dataset/`.

4\. Ensure the engine binaries and weights referenced by the scripts exist at the expected paths.

5\. Run the scripts from the appropriate folder or adapt the paths to your local setup.



\## Reproducibility

The repository is meant to show both the thesis results and the workflow used to produce them.  

The included scripts and example outputs should make it easier to understand how the experiments were run and how the final figures were generated.

