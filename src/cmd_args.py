
from types import SimpleNamespace
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", default="src/experiments/exp1.yaml", help="YAML configuration file")


def parse_args():
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    opts = SimpleNamespace(**configs)
    return opts
