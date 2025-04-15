
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def set_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_logger(log_file="out.log"):
    import logging
    from rich.logging import RichHandler

    os.makedirs("logs", exist_ok=True)  # logs directory
    log_file = os.path.join("logs", log_file)  # log path

    # Create handlers
    rich_handler = RichHandler(rich_tracebacks=True)
    file_handler = logging.FileHandler(log_file)
    
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
        self.reset()

    def reset(self):
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def update(self, val, n=1):
        # update statistic with given new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.size == y_pred.size
    squared_residuals = la.norm(y_true - y_pred) ** 2
    sum_of_y_mean = la.norm(y_true - np.mean(y_true)) ** 2
    return 1 - squared_residuals / sum_of_y_mean
    # adjusted version
    # return 1 - (squared_residuals / (n - p)) / (sum_of_y / (n - 1))


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.size == y_pred.size
    n_samples = y_true.size
    squared_residuals = la.norm(y_true - y_pred) ** 2
    return np.sqrt(squared_residuals / n_samples)


def plot_metric(vals, title, output_dir):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(vals) + 1), vals)
    # ax.set_yscale("log")
    # ax.set_ylim(bottom=1e-2)
    ax.set_title(title)
    ax.set_xlabel("iters")
    ax.grid("both")

    path = "plots"
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, output_dir))
