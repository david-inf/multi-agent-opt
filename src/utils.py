
import os
import random
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import logging
from rich.logging import RichHandler


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_logger(log_file="out.log"):
    output_dir = "src/logs"
    os.makedirs(output_dir, exist_ok=True)  # logs directory
    output_path = os.path.join(output_dir, log_file)  # log path

    # Create handlers
    rich_handler = RichHandler(rich_tracebacks=True)
    file_handler = logging.FileHandler(output_path)
    
    # Set the same format for both handlers
    FORMAT = "%(message)s"
    # formatter = logging.Formatter(FORMAT, datefmt="[%X]")
    # file_handler.setFormatter(formatter)
    # rich_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", 
                        handlers=[rich_handler, file_handler])
    
    log = logging.getLogger("rich")
    return log

LOG = get_logger()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        """Initialize the AverageMeter with default values."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def reset(self):
        """Reset all statistics to zero."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def update(self, val, n=1):
        """Update statistics with new value.
        
        Args:
            val: The value to update with
            n: Weight of the value (default: 1)
        """
        # update statistic with given new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 goodness of fit"""
    assert y_true.size == y_pred.size
    squared_residuals = la.norm(y_true - y_pred) ** 2
    sum_of_y_mean = la.norm(y_true - np.mean(y_true)) ** 2
    return 1 - squared_residuals / sum_of_y_mean
    # adjusted version
    # return 1 - (squared_residuals / (n - p)) / (sum_of_y / (n - 1))


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """RMSE - root mean squared error"""
    assert y_true.size == y_pred.size
    n_samples = y_true.size
    squared_residuals = la.norm(y_true - y_pred) ** 2
    return np.sqrt(squared_residuals / n_samples)


def plot_metric(vals, output_path):
    """Use for convergence objective against iterations"""
    _, ax = plt.subplots()
    ax.plot(np.arange(len(vals)), vals)
    # ax.set_yscale("log")
    # ax.set_ylim(bottom=1e-2)
    ax.set_title("Consensus convergence")
    ax.set_xlabel("iters")
    ax.set_ylabel("Consensus error")
    ax.set_yscale("log")
    ax.grid("both")

    plt.savefig(output_path)


def plot_alpha(vals_mat: np.ndarray, ylab, title, groundtruth: float, output_path):
    """Use for parameters convergence"""
    _, ax = plt.subplots()
    iters = vals_mat.shape[0]
    for i in range(vals_mat.shape[1]):
        ax.plot(np.arange(iters), vals_mat[:, i], label=f"Agent {i}")

    ax.set_title(title)
    ax.set_xlabel("iters")
    ax.set_ylabel(ylab)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")
    if groundtruth:
        ax.axhline(y=groundtruth, color="r", linestyle="--", linewidth=2)
        ax.set_ylim(groundtruth-0.1, groundtruth+0.1)

    plt.savefig(output_path)


def plot_beta(vals_mat: np.ndarray, ylab, title, groundtruth: np.ndarray, output_path):
    """Use for parameters convergence"""
    _, ax = plt.subplots()
    iters = vals_mat.shape[0]
    for i in range(vals_mat.shape[1]):
        ax.plot(np.arange(iters), abs(vals_mat[:, i]-groundtruth[i]), label=f"Agent {i}")

    ax.set_title(title)
    ax.set_xlabel("iters")
    ax.set_ylabel(ylab)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_ylim(0, 0.2)
    ax.legend(loc="upper right")
    ax.axhline(y=0., color="r", linestyle="--", linewidth=2)

    plt.savefig(output_path)
