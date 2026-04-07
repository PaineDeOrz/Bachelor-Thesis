# run_train.py (in repo root)
import yaml
from utils import parse_args  # Absolute import works here
from train import run

if __name__ == "__main__":
    cfg = parse_args('config.yaml')
    run(cfg)
