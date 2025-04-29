
from types import SimpleNamespace
import argparse
import yaml
# from utils import get_logger


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="src/configs/balanced_random5.yaml",
                    help="YAML configuration file")


def parse_args():
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    opts = SimpleNamespace(**configs)

    # LOG = get_logger(opts.experiment_name + ".log")

    return opts
