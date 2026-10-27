"""Count real-sized model parameters without allocating their storage."""

import argparse
import json

import torch

from .config import load_config
from .model import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wan2.1_1.3b.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    with torch.device("meta"):
        model = create_model(cfg, allow_random=True)
    print(json.dumps(model.parameter_counts(), indent=2))


if __name__ == "__main__":
    main()
